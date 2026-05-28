"""
Offline helper for automated calculation of `pupil_samples`.

This module implements a one-time, high-resolution procedure to determine the
minimum required grid size for the pupil function (`pupil_samples`) that
ensures the resulting PSF is not effectively truncated due to finite grid
support.

Design & Usage
--------------
- This computation is **not** part of the main simulation pipeline. It is
  intended to be run manually (or as part of an offline calibration script)
  once for each optical preset (combination of NA, wavelength, medium index,
  apodization, etc.).
- The resulting integer `pupil_samples` is then written into the preset
  definition (e.g., in `config.PARAMS` or an instrument preset), and the main
  simulation uses that fixed value.

Algorithm
---------
Inputs:
    - numerical_aperture (NA)
    - wavelength_nm
    - refractive_index_medium
    - apodization_factor
    - amplitude_threshold (relative to peak amplitude, e.g., 1e-6)
    - ground_truth_samples (e.g., 65,536)

Steps:
    1. Construct a high-resolution 1D pupil function P(u) representing a
       circular aperture with apodization:
           - u is a dimensionless radial coordinate in the pupil plane.
           - |u| <= 1 corresponds to the open aperture; outside this region,
             P(u) = 0.
           - Inside the aperture, the amplitude is modulated by a simple
             Gaussian-like apodization exp(-apodization_factor * rho^2),
             where rho = |u|.

    2. Compute the 1D complex Amplitude Spread Function (ASF) via a large 1D
       FFT:
           ASF(x) = FFT{P(u)} (up to constant factors).
       This 1D ASF is a radial cross-section of the full 2D PSF for a
       radially symmetric pupil.

    3. Normalize the ASF amplitude so that its peak magnitude is 1.0:
           A(x) = |ASF(x)| / max_x |ASF(x)|

    4. Scan radially outward from the central peak and record the smallest
       radius (in samples) at which A(x) first falls below the chosen
       amplitude_threshold. This radius is r_cutoff.

    5. The minimum required PSF diameter in samples is:
           D_required = 2 * r_cutoff

    6. For FFT efficiency in the main simulation, choose the smallest power of
       two greater than or equal to D_required:
           pupil_samples = 2**ceil(log2(D_required))

This procedure works in dimensionless pupil coordinates. NA, wavelength, and
refractive index are validated, while the normalized ASF shape is governed by
the aperture and apodization.

Note:
    - This implementation deliberately avoids any dependency on the rest of the
      simulation pipeline. It is safe to run standalone and does not modify
      global state or PARAMS.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def compute_pupil_samples(
    numerical_aperture: float,
    wavelength_nm: float,
    refractive_index_medium: float,
    apodization_factor: float,
    amplitude_threshold: float = 1e-6,
    ground_truth_samples: int = 65536,
) -> int:
    """
    Compute a recommended value for `pupil_samples` using a high-resolution
    1D ASF calculation.

    This function is designed for offline use. It accepts the key optical
    parameters, constructs a dimensionless 1D pupil function with apodization,
    computes a high-resolution ASF via FFT, and then determines the minimum
    required pupil grid size so that the ASF is negligible (below
    amplitude_threshold) at the grid edge.

    Args:
        numerical_aperture (float):
            Objective numerical aperture (NA). Must satisfy
                0 < NA <= refractive_index_medium.
            Used here for basic validation; the ASF calculation is performed
            in normalized coordinates and does not depend on the absolute
            scaling of NA.

        wavelength_nm (float):
            Illumination wavelength in vacuum in nanometers. Must be positive.
            Used for validation; the 1D ASF calculation is performed in
            normalized coordinates.

        refractive_index_medium (float):
            Refractive index of the sample medium. Must be positive. NA is
            checked against this value.

        apodization_factor (float):
            Dimensionless factor controlling the strength of Gaussian-like
            apodization in the pupil amplitude. Larger values increase the
            attenuation near the pupil edge. This is the same parameter used
            in the main simulation's pupil function.

        amplitude_threshold (float, optional):
            Relative amplitude threshold in (0, 1). The ASF amplitude is
            normalized so that its peak value is 1.0; the algorithm identifies
            the first radius where the amplitude falls below this value and
            uses that as the cutoff. A typical value is 1e-6.

        ground_truth_samples (int, optional):
            Number of samples in the high-resolution 1D pupil and ASF used to
            approximate the "ground truth" PSF. Must be a reasonably large
            positive integer. A typical value is 65,536 (2**16). The actual
            FFT length used is rounded up to the next power of two.

    Returns:
        int:
            Recommended `pupil_samples` value, guaranteed to be a power of two
            and at least 4. This value can be used directly in the main
            simulation for the given optical configuration.

    Raises:
        ValueError:
            If any of the input parameters are invalid (e.g., non-positive NA,
            wavelength, or medium index; amplitude_threshold not in (0, 1);
            ground_truth_samples too small).
    """
    # --- Basic input validation ---
    NA = float(numerical_aperture)
    wavelength_nm = float(wavelength_nm)
    n_medium = float(refractive_index_medium)
    apodization_factor = float(apodization_factor)
    amplitude_threshold = float(amplitude_threshold)
    ground_truth_samples = int(ground_truth_samples)

    if NA <= 0.0:
        raise ValueError("numerical_aperture must be positive.")
    if n_medium <= 0.0:
        raise ValueError("refractive_index_medium must be positive.")
    if NA > n_medium:
        raise ValueError(
            "numerical_aperture must not exceed refractive_index_medium. "
            f"Got NA={NA}, n_medium={n_medium}."
        )
    if wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be positive.")
    if not (0.0 < amplitude_threshold < 1.0):
        raise ValueError(
            "amplitude_threshold must be in the open interval (0, 1). "
            f"Got {amplitude_threshold}."
        )
    if ground_truth_samples < 16:
        raise ValueError(
            "ground_truth_samples must be at least 16 for a meaningful ASF "
            f"calculation. Got {ground_truth_samples}."
        )

    # Round the ground-truth FFT length up to the next power of two to ensure
    # good FFT performance and symmetric grids.
    N_gt_power = int(math.ceil(math.log2(ground_truth_samples)))
    N_gt = 1 << N_gt_power

    # --- Construct the high-resolution 1D pupil function ---
    # We work in a dimensionless coordinate u in [-1, 1), where |u| <= 1
    # corresponds to the open circular aperture. The radial coordinate is
    # rho = |u|. Inside the aperture, we apply a Gaussian-like apodization
    # exp(-apodization_factor * rho^2). Outside, the pupil amplitude is zero.
    #
    # This yields a radially symmetric pupil in the 2D problem; here we work
    # with a 1D cross-section sufficient for determining the radial extent of
    # the ASF.
    u = np.linspace(-1.0, 1.0, N_gt, endpoint=False)
    rho = np.abs(u)

    aperture = (rho <= 1.0).astype(np.float64)
    if apodization_factor != 0.0:
        apodization = np.exp(-apodization_factor * rho * rho)
    else:
        apodization = np.ones_like(rho, dtype=np.float64)

    pupil = aperture * apodization

    # --- Compute the "ground truth" 1D ASF via FFT ---
    # We compute the inverse transform (up to constants) of the pupil function,
    # centering both the pupil and the ASF using fftshift/ifftshift so that
    # the ASF peak is located at the central index.
    #
    # The absolute scaling of the coordinate does not matter here; only the
    # relative decay of the ASF amplitude with radius is needed.
    asf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pupil)))
    amplitude = np.abs(asf)

    # Normalize so that the maximum amplitude is exactly 1.0.
    max_amp = float(amplitude.max())
    if max_amp <= 0.0:
        # Degenerate case: if the ASF is numerically zero everywhere, fall
        # back to a small default pupil size.
        return 64

    amplitude /= max_amp

    # --- Find the cutoff radius where amplitude falls below the threshold ---
    # We treat the central index as the origin and scan outward to find the
    # smallest radius r_cutoff (in samples) where the normalized amplitude
    # first drops below amplitude_threshold. This is a conservative criterion:
    # the ASF amplitude may oscillate, but for a sufficiently small threshold
    # (e.g., 1e-6), the first crossing effectively bounds the region of
    # physically relevant signal.
    center = N_gt // 2

    # Radial profile from the center to the edge on one side.
    radial_profile = amplitude[center:]

    below = radial_profile < amplitude_threshold
    if not np.any(below):
        # The amplitude never falls below the threshold within the available
        # grid. In this extreme case, use the maximum possible radius.
        r_cutoff = len(radial_profile) - 1
    else:
        # Index of the first sample where amplitude < threshold.
        first_below_idx = int(np.argmax(below))
        r_cutoff = first_below_idx

    # Minimum required diameter in samples to contain the ASF above threshold.
    D_required = max(2 * r_cutoff, 4)

    # --- Choose the nearest power-of-two pupil_samples >= D_required ---
    pupil_power = int(math.ceil(math.log2(D_required)))
    pupil_samples = 1 << pupil_power

    # Ensure a lower bound to avoid pathological tiny grids.
    pupil_samples = max(pupil_samples, 16)

    return pupil_samples


def recommend_pupil_samples_for_params(
    params: Dict[str, object],
    amplitude_threshold: float = 1e-6,
    ground_truth_samples: int = 65536,
) -> int:
    """
    Convenience wrapper that computes a recommended `pupil_samples` value
    directly from a PARAMS-style dictionary.

    This function reads the relevant optical parameters from `params`:

        - "numerical_aperture"
        - "wavelength_nm"
        - "refractive_index_medium"
        - "apodization_factor"

    and passes them to `compute_pupil_samples`. It does **not** modify the
    input dictionary.

    Example:
        >>> from config import PARAMS
        >>> from pupil_sampling import recommend_pupil_samples_for_params
        >>> ps = recommend_pupil_samples_for_params(PARAMS)
        >>> print(ps)

    Args:
        params (dict):
            Simulation parameter dictionary (e.g., `config.PARAMS`). Must
            contain the keys listed above.

        amplitude_threshold (float, optional):
            Relative amplitude threshold in (0, 1) used by the underlying
            `compute_pupil_samples` function. Defaults to 1e-6.

        ground_truth_samples (int, optional):
            High-resolution FFT length used by the underlying
            `compute_pupil_samples` function. Defaults to 65,536.

    Returns:
        int:
            Recommended `pupil_samples` for the given optical configuration.

    Raises:
        KeyError:
            If any required keys are missing from `params`.
        ValueError:
            If any of the parameter values are invalid.
    """
    try:
        NA = float(params["numerical_aperture"])
        wavelength_nm = float(params["wavelength_nm"])
        n_medium = float(params["refractive_index_medium"])
        apodization_factor = float(params["apodization_factor"])
    except KeyError as exc:
        raise KeyError(
            f"Missing required key in params for pupil_samples recommendation: {exc}"
        ) from exc

    return compute_pupil_samples(
        numerical_aperture=NA,
        wavelength_nm=wavelength_nm,
        refractive_index_medium=n_medium,
        apodization_factor=apodization_factor,
        amplitude_threshold=amplitude_threshold,
        ground_truth_samples=ground_truth_samples,
    )


if __name__ == "__main__":
    # Command-line entry point for quick offline usage.
    #
    # When invoked as a script, this will:
    #   - Import the global PARAMS from config.py.
    #   - Compute a recommended pupil_samples value for those parameters.
    #   - Print the result alongside the current PARAMS["pupil_samples"] value
    #     for inspection.
    #
    # This does not modify PARAMS or any preset definitions.
    import argparse

    from config import PARAMS

    parser = argparse.ArgumentParser(
        description=(
            "Offline helper to compute a recommended `pupil_samples` value "
            "for the current optical configuration."
        )
    )
    parser.add_argument(
        "--amplitude_threshold",
        type=float,
        default=1e-6,
        help=(
            "Relative ASF amplitude threshold in (0, 1) used to define the "
            "cutoff radius (default: 1e-6)."
        ),
    )
    parser.add_argument(
        "--ground_truth_samples",
        type=int,
        default=65536,
        help=(
            "Number of samples in the high-resolution 1D pupil/ASF grid used "
            "for the offline calculation (default: 65536)."
        ),
    )
    args = parser.parse_args()

    recommended = recommend_pupil_samples_for_params(
        PARAMS,
        amplitude_threshold=args.amplitude_threshold,
        ground_truth_samples=args.ground_truth_samples,
    )

    current = PARAMS.get("pupil_samples", None)

    print("=== Pupil Samples Recommendation ===")
    print(f"Current PARAMS['pupil_samples']: {current!r}")
    print(f"Recommended pupil_samples       : {recommended}")
