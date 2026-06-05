#!/usr/bin/env python3
"""Validate modality contrast equations and documented limiting cases."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from config import PARAMS  # noqa: E402
from imaging_models.coherent_brightfield import CoherentBrightfieldImagingModel  # noqa: E402
from imaging_models.coherent_darkfield import CoherentDarkFieldImagingModel  # noqa: E402
from imaging_models.electron_constants import electron_wavelength_m, scherzer_defocus_m  # noqa: E402
from imaging_models.fluorescence_tirf import TIRFFluorescenceImagingModel  # noqa: E402
from imaging_models.fluorescence_widefield import FluorescenceWidefieldImagingModel  # noqa: E402
from imaging_models.kohler import PartiallyCoherentBrightfieldImagingModel  # noqa: E402
from imaging_models.off_axis_holography import OffAxisHolographyImagingModel  # noqa: E402
from imaging_models.qpi import QuantitativePhaseImagingModel  # noqa: E402
from imaging_models.ricm import ReflectionInterferenceContrastImagingModel  # noqa: E402
from imaging_models.tem_backends.ctf_proxy import CTFProxyTEMBackend  # noqa: E402
from imaging_models.zernike_phase import ZernikePhaseContrastImagingModel  # noqa: E402
from modality_profiles import profile_card_for_model  # noqa: E402


def _params(**updates):
    params = deepcopy(PARAMS)
    params.update(updates)
    return params


def _print_result(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def check_darkfield_equation() -> bool:
    rng = np.random.default_rng(3)
    e_sca = (rng.standard_normal((48, 48)) + 1j * rng.standard_normal((48, 48))) * 0.05
    background = np.ones((48, 48), dtype=np.complex128)
    errors = []
    ref_independent = True
    for gain in (1.0, 3.0):
        params = _params(imaging_model="coherent_dark_field", dark_field_field_gain=gain)
        model = CoherentDarkFieldImagingModel(params)
        contrast = model.compute_per_particle_contrast(e_sca, background, params)
        expected = (gain * gain) * np.abs(e_sca) ** 2
        errors.append(float(np.max(np.abs(contrast - expected))))
        changed_reference = model.compute_per_particle_contrast(e_sca, background * 7.0 + 3j, params)
        ref_independent = ref_independent and bool(np.allclose(contrast, changed_reference))
    max_error = max(errors)
    return _print_result(
        "darkfield_intensity_equation",
        max_error <= 1.0e-12 and ref_independent,
        f"max_abs_error={max_error:.3e} ref_independent={ref_independent}",
    )


def check_qpi_phase_equation() -> bool:
    rng = np.random.default_rng(4)
    e_sca = (rng.standard_normal((48, 48)) + 1j * rng.standard_normal((48, 48))) * 0.05
    background = np.ones((48, 48), dtype=np.complex128)
    params = _params(imaging_model="quantitative_phase", reference_field_amplitude=1.0)
    model = QuantitativePhaseImagingModel(params)
    phase = model.compute_per_particle_contrast(e_sca, background, params)
    expected = np.angle(1.0 + e_sca / background)
    identity_error = float(np.max(np.abs(phase - expected)))
    tiny = e_sca * 1.0e-3
    small_phase = model.compute_per_particle_contrast(tiny, background, params)
    small_expected = np.imag(tiny / background)
    small_error = float(np.max(np.abs(small_phase - small_expected)))
    return _print_result(
        "qpi_phase_equation",
        identity_error <= 1.0e-12 and small_error <= 1.0e-6,
        f"identity_error={identity_error:.3e} small_signal_error={small_error:.3e}",
    )


def check_ricm_interference_equation() -> bool:
    rng = np.random.default_rng(5)
    e_sca = (rng.standard_normal((48, 48)) + 1j * rng.standard_normal((48, 48))) * 0.05
    background = np.ones((48, 48), dtype=np.complex128)
    params = _params(
        imaging_model="ricm",
        ricm_wavelength_nm=550.0,
        ricm_particle_reflection_model="param",
        ricm_particle_reflection_coefficient=0.04,
        ricm_interface_phase_shift_rad=np.pi,
    )
    model = ReflectionInterferenceContrastImagingModel(params)
    contrast = model.compute_per_particle_contrast(e_sca, background, params)
    reference = background * complex(model._r_s)
    prefactor = complex(model._sca_prefactor())
    expected = np.abs(reference + prefactor * e_sca) ** 2 - np.abs(reference) ** 2
    error = float(np.max(np.abs(contrast - expected)))
    return _print_result("ricm_interference_equation", error <= 1.0e-12, f"max_abs_error={error:.3e}")


def check_off_axis_holography_equation() -> bool:
    rng = np.random.default_rng(6)
    shape = (64, 64)
    e_sca = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 0.05
    background = np.ones(shape, dtype=np.complex128)
    period_px = 8.0
    params = _params(
        imaging_model="off_axis_holography",
        off_axis_fringe_period_px=period_px,
        off_axis_fringe_angle_rad=0.0,
        psf_oversampling_factor=1,
    )
    model = OffAxisHolographyImagingModel(params)
    carrier = model._tilt_field(shape)
    contrast = model.compute_per_particle_contrast(e_sca, background, params)
    object_field = background
    reference = model._reference_amplitude * model._reference_amplitude_scale * carrier
    empty_frame = object_field + reference
    expected = np.abs(empty_frame + e_sca) ** 2 - np.abs(empty_frame) ** 2
    identity_error = float(np.max(np.abs(contrast - expected)))

    uniform_sample = np.full(shape, 0.1 + 0.0j)
    intensity = model.compute_intensity(uniform_sample, background, params)
    row = intensity[shape[0] // 2] - float(np.mean(intensity[shape[0] // 2]))
    spectrum = np.abs(np.fft.rfft(row))
    peak_bin = int(np.argmax(spectrum[1:]) + 1)
    measured_period = shape[1] / peak_bin
    period_error = abs(measured_period / model._period_canvas_px - 1.0)
    return _print_result(
        "off_axis_holography_equation",
        identity_error <= 1.0e-12 and period_error <= 1.0e-12,
        (
            f"identity_error={identity_error:.3e} "
            f"measured_canvas_period_px={measured_period:.3f} "
            f"expected_canvas_period_px={model._period_canvas_px:.3f}"
        ),
    )


def check_fluorescence_widefield_psf() -> bool:
    sigma_px = 3.0
    params = _params(
        imaging_model="fluorescence_widefield",
        fluorescence_backend="parametric_psf",
        fluorescence_emission_psf_sigma_px=sigma_px,
        fluorescence_emission_psf_sigma_nm=None,
        psf_oversampling_factor=1,
        pixel_size_nm=100.0,
    )
    model = FluorescenceWidefieldImagingModel(params)
    size = 81
    source = np.zeros((size, size), dtype=float)
    center = size // 2
    source[center, center] = 1.0
    blurred = model._emission_blur(source)
    yy, xx = np.indices((size, size), dtype=float)
    r2 = (xx - center) ** 2 + (yy - center) ** 2
    measured_sigma = float(np.sqrt(np.sum(blurred * r2) / np.sum(blurred) / 2.0))
    sigma_error = abs(measured_sigma / sigma_px - 1.0)
    flux_error = abs(float(np.sum(blurred)) - 1.0)
    a = np.zeros((size, size), dtype=float)
    b = np.zeros((size, size), dtype=float)
    a[20, 20] = 1.0
    b[60, 60] = 2.0
    linear_error = float(np.max(np.abs(model._emission_blur(a + b) - model._emission_blur(a) - model._emission_blur(b))))
    return _print_result(
        "fluorescence_widefield_parametric_psf",
        sigma_error <= 0.02 and flux_error <= 1.0e-12 and linear_error <= 1.0e-12,
        (
            f"measured_sigma_px={measured_sigma:.3f} expected_sigma_px={sigma_px:.3f} "
            f"sigma_rel_error={sigma_error:.3e} flux_error={flux_error:.3e} "
            f"linearity_error={linear_error:.3e}"
        ),
    )


def check_fluorescence_count_scaling_contract() -> bool:
    params = _params(
        imaging_model="fluorescence_widefield",
        fluorescence_backend="parametric_psf",
        fluorescence_photons_per_fluorophore_per_frame=120.0,
        fluorescence_collection_efficiency=0.25,
        fluorescence_detector_qe=0.8,
    )
    model = FluorescenceWidefieldImagingModel(params)
    response = model.compute_response_function((32, 32), params)
    card = profile_card_for_model(params, model, response_function=response, model_canvas_shape=(32, 32))
    expected_mode = "physical_fluorophore_photon_budget"
    expected_scale = 120.0 * 0.25 * 0.8
    response_mode = response.get("count_scaling_mode")
    card_mode = card.get("count_scaling_mode")
    response_scale = float(response.get("count_scale", np.nan))
    duplicate_namespaced_keys = sorted(
        key for key in ("fluorescence_count_scale", "fluorescence_count_scaling_mode") if key in response
    )
    ok = (
        response_mode == expected_mode
        and card_mode == expected_mode
        and abs(response_scale / expected_scale - 1.0) <= 1.0e-12
        and not duplicate_namespaced_keys
    )
    return _print_result(
        "fluorescence_count_scaling_contract",
        ok,
        (
            f"response_mode={response_mode!r} profile_mode={card_mode!r} "
            f"count_scale={response_scale:.6g} expected={expected_scale:.6g} "
            f"duplicate_namespaced_keys={duplicate_namespaced_keys}"
        ),
    )


def check_tirf_penetration_depth() -> bool:
    wavelength_nm = 488.0
    n_prism = 1.518
    n_sample = 1.33
    angle_deg = 70.0
    params = _params(
        imaging_model="fluorescence_tirf",
        fluorescence_backend="parametric_psf",
        tirf_use_angle_derived_penetration_depth=True,
        fluorescence_excitation_wavelength_nm=wavelength_nm,
        tirf_prism_refractive_index=n_prism,
        tirf_sample_refractive_index=n_sample,
        tirf_incident_angle_deg=angle_deg,
    )
    observed = TIRFFluorescenceImagingModel.penetration_depth_nm(params)
    expected = wavelength_nm / (
        4.0
        * np.pi
        * np.sqrt((n_prism * np.sin(np.deg2rad(angle_deg))) ** 2 - n_sample ** 2)
    )
    rel_error = abs(observed / expected - 1.0)
    rejected = False
    try:
        TIRFFluorescenceImagingModel.penetration_depth_nm(
            _params(
                imaging_model="fluorescence_tirf",
                fluorescence_backend="parametric_psf",
                tirf_use_angle_derived_penetration_depth=True,
                fluorescence_excitation_wavelength_nm=wavelength_nm,
                tirf_prism_refractive_index=n_prism,
                tirf_sample_refractive_index=n_sample,
                tirf_incident_angle_deg=45.0,
            )
        )
    except ValueError:
        rejected = True
    return _print_result(
        "tirf_penetration_depth",
        rel_error <= 1.0e-12 and rejected,
        f"relative_error={rel_error:.3e} below_critical_rejected={rejected}",
    )


def check_zernike_phase_visibility() -> bool:
    size = 64
    center = size // 2
    yy, xx = np.indices((size, size), dtype=float)
    phase_object = 0.15 * np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * 6.0 ** 2))
    e_sca = np.exp(1j * phase_object) - 1.0

    def contrast(shift: float):
        params = _params(
            imaging_model="zernike_phase",
            reference_field_amplitude=1.0,
            zernike_model="fourier_phase_ring_proxy",
            zernike_phase_ring_inner_fraction=0.0,
            zernike_phase_ring_outer_fraction=0.08,
            zernike_phase_ring_shift_rad=shift,
            zernike_phase_ring_amplitude=1.0,
        )
        return ZernikePhaseContrastImagingModel(params).compute_per_particle_contrast(e_sca, None, params)

    with_ring = contrast(np.pi / 2.0)
    no_ring = contrast(0.0)
    ratio = float(np.linalg.norm(with_ring) / max(float(np.linalg.norm(no_ring)), 1.0e-30))
    return _print_result("zernike_phase_visibility", ratio >= 1.0e6, f"ring_to_no_ring_norm_ratio={ratio:.3e}")


def check_tem_ctf_proxy() -> bool:
    kv = 300.0
    cs_mm = 1.0
    wavelength_m = electron_wavelength_m(kv)
    cs_m = 1.0e-3 * cs_mm
    defocus_m = scherzer_defocus_m(kv, cs_mm)
    pixel_size_m = 5.0e-11
    backend = CTFProxyTEMBackend(
        pixel_size_m=pixel_size_m,
        electron_wavelength_m=wavelength_m,
        Cs_mm=cs_mm,
        defocus_m=defocus_m,
        partial_coherence_alpha_mrad=0.0,
    )
    shape = (256, 256)
    ctf = backend.ctf(shape)
    fx = np.fft.fftfreq(shape[1], d=pixel_size_m)
    fy = np.fft.fftfreq(shape[0], d=pixel_size_m)
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    k = np.sqrt(kx * kx + ky * ky)
    chi = (np.pi * wavelength_m ** 3 * cs_m * 0.5) * k ** 4 - (np.pi * wavelength_m * defocus_m) * k ** 2
    expected = 2.0 * np.sin(chi)
    identity_error = float(np.max(np.abs(ctf - expected)))

    positive_k = np.sort(fx[fx > 0.0])
    chi_axis = (
        (np.pi * wavelength_m ** 3 * cs_m * 0.5) * positive_k ** 4
        - (np.pi * wavelength_m * defocus_m) * positive_k ** 2
    )
    ctf_axis = 2.0 * np.sin(chi_axis)
    zero_k = None
    for idx in range(1, len(ctf_axis)):
        if ctf_axis[idx - 1] != 0.0 and np.sign(ctf_axis[idx]) != np.sign(ctf_axis[idx - 1]):
            zero_k = positive_k[idx]
            break
    measured_resolution_m = np.inf if zero_k is None else 1.0 / zero_k
    scherzer_resolution_m = 0.66 * (cs_m ** 0.25) * (wavelength_m ** 0.75)
    resolution_rel_error = abs(measured_resolution_m / scherzer_resolution_m - 1.0)
    aperture_mrad = 5.0
    aperture_backend = CTFProxyTEMBackend(
        pixel_size_m=pixel_size_m,
        electron_wavelength_m=wavelength_m,
        Cs_mm=cs_mm,
        defocus_m=defocus_m,
        partial_coherence_alpha_mrad=0.0,
        objective_aperture_mrad=aperture_mrad,
    )
    aperture_ctf = aperture_backend.ctf(shape)
    k_max = (1.0e-3 * aperture_mrad) / wavelength_m
    clipped = bool(np.allclose(aperture_ctf[k > k_max], 0.0))
    interior_matches = bool(np.allclose(aperture_ctf[k <= k_max], expected[k <= k_max]))
    return _print_result(
        "tem_ctf_proxy",
        identity_error <= 1.0e-12 and resolution_rel_error <= 0.06 and clipped and interior_matches,
        (
            f"ctf_identity_error={identity_error:.3e} "
            f"first_zero_resolution_rel_error={resolution_rel_error:.3e} "
            f"objective_aperture_clipped={clipped} objective_aperture_interior_matches={interior_matches}"
        ),
    )


def check_kohler_coherent_limit() -> bool:
    rng = np.random.default_rng(7)
    shape = (64, 64)
    pixel_size_nm = 100.0
    wavelength_nm = 550.0
    numerical_aperture = 0.6
    random_field = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 0.03
    spectrum = np.fft.fft2(random_field)
    fx = np.fft.fftfreq(shape[1], d=pixel_size_nm * 1.0e-9)
    fy = np.fft.fftfreq(shape[0], d=pixel_size_nm * 1.0e-9)
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    pupil = (kx * kx + ky * ky) <= (numerical_aperture / (wavelength_nm * 1.0e-9)) ** 2
    bandlimited_field = np.fft.ifft2(spectrum * pupil)
    background = np.ones(shape, dtype=np.complex128)
    common = dict(
        reference_field_amplitude=1.0,
        numerical_aperture=numerical_aperture,
        refractive_index_medium=1.0,
        wavelength_nm=wavelength_nm,
        pixel_size_nm=pixel_size_nm,
        psf_oversampling_factor=1,
    )
    kohler_params = _params(
        imaging_model="partially_coherent_bright_field",
        kohler_source_samples=1,
        kohler_coherence_factor=0.0,
        **common,
    )
    coherent_params = _params(imaging_model="coherent_bright_field", **common)
    kohler = PartiallyCoherentBrightfieldImagingModel(kohler_params).compute_per_particle_contrast(
        bandlimited_field,
        background,
        kohler_params,
    )
    coherent = CoherentBrightfieldImagingModel(coherent_params).compute_per_particle_contrast(
        bandlimited_field,
        background,
        coherent_params,
    )
    abs_error = float(np.max(np.abs(kohler - coherent)))
    rel_error = abs_error / max(float(np.max(np.abs(coherent))), 1.0e-30)
    return _print_result(
        "kohler_bandlimited_coherent_limit",
        rel_error <= 1.0e-9,
        f"max_abs_error={abs_error:.3e} relative_error={rel_error:.3e}",
    )


def run_validation() -> bool:
    checks = [
        check_darkfield_equation,
        check_qpi_phase_equation,
        check_ricm_interference_equation,
        check_off_axis_holography_equation,
        check_fluorescence_widefield_psf,
        check_fluorescence_count_scaling_contract,
        check_tirf_penetration_depth,
        check_zernike_phase_visibility,
        check_tem_ctf_proxy,
        check_kohler_coherent_limit,
    ]
    results = [check() for check in checks]
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if run_validation() else 1)
