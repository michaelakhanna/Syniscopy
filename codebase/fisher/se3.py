"""SE(3), orientation, and symmetry-rank Fisher diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from common_utils import init_infinite_dict
from shared_constants import SE3_STATE_AXES

from ._constants import (
    _FISHER_RANGE_RESIDUAL_TOL,
    _FISHER_RANK_RELATIVE_TOL,
    _FISHER_SE3_AXIS_RELATIVE_TOL,
    _FISHER_SE3_EPS,
)
from ._metadata_helpers import (
    _derivative_unit,
    _diagnostic_metadata_aliases,
    _fisher_rank_metadata,
    _variance_units,
)

def _se3_derivative_metadata(
    pixel_size_nm: float,
    z_step_nm: float,
    rotation_step_rad: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """Metadata for the mixed-unit SE(3) Fisher derivative convention."""
    state_axes = ["x", "y", "z", "omega_x", "omega_y", "omega_z"]
    state_axis_units = {
        "x": "nm",
        "y": "nm",
        "z": "nm",
        "omega_x": "radian",
        "omega_y": "radian",
        "omega_z": "radian",
    }
    derivative_units = [
        _derivative_unit(signal_units, state_axis_units[axis])
        for axis in state_axes
    ]
    variance_units = (
        str(noise_variance_units)
        if noise_variance_units is not None
        else _variance_units(signal_units)
    )
    return {
        "state_axes": state_axes,
        "measurement_domain": str(measurement_domain),
        "signal_units": str(signal_units),
        "state_axis_units": state_axis_units,
        "derivative_units": derivative_units,
        "lateral_derivative_mode": "detector_grid_central_difference_stationary_shift",
        "axial_derivative_mode": "symmetric_rerendered_z_pair",
        "orientation_derivative_mode": "symmetric_body_frame_rotation_pair",
        "pixel_size_nm": float(pixel_size_nm),
        "z_step_nm": float(z_step_nm),
        "rotation_step_rad": float(rotation_step_rad),
        "noise_variance_units": variance_units,
        "fisher_axis_units": [
            "1/nm^2",
            "1/nm^2",
            "1/nm^2",
            "1/rad^2",
            "1/rad^2",
            "1/rad^2",
        ],
        "fisher_units_by_entry": {
            axis_i: {
                axis_j: f"{derivative_units[i]}*{derivative_units[j]}/{variance_units}"
                for j, axis_j in enumerate(state_axes)
            }
            for i, axis_i in enumerate(state_axes)
        },
        "axis_relative_tolerance": float(_FISHER_SE3_AXIS_RELATIVE_TOL),
        "relative_rank_tolerance": float(_FISHER_RANK_RELATIVE_TOL),
        "range_residual_tolerance": float(_FISHER_RANGE_RESIDUAL_TOL),
    }

def _validate_se3_renders(
    renders: dict[str, np.ndarray],
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """
    Check that ``renders`` contains the nine required keys with consistent
    shape. The "centre" render is treated as the in-focus, in-orientation
    reference; every other render is a one-axis perturbation.
    """
    required_keys = {
        "centre",
        "z_minus", "z_plus",
        "rx_minus", "rx_plus",
        "ry_minus", "ry_plus",
        "rz_minus", "rz_plus",
    }
    missing = required_keys - set(renders)
    if missing:
        raise ValueError(
            f"compute_fisher_information_se3 requires keys "
            f"{sorted(required_keys)}; missing: {sorted(missing)}."
        )
    centre = np.asarray(renders["centre"], dtype=float)
    if centre.ndim != 2:
        raise ValueError(
            f"renders['centre'] must be a 2D image; got shape {centre.shape}."
        )
    if not np.all(np.isfinite(centre)):
        raise ValueError("renders['centre'] must contain only finite values.")
    if expected_shape is not None and centre.shape != expected_shape:
        raise ValueError(
            f"renders['centre'] shape {centre.shape} does not match expected "
            f"{expected_shape}."
        )
    for k in required_keys - {"centre"}:
        arr = np.asarray(renders[k], dtype=float)
        if arr.shape != centre.shape:
            raise ValueError(
                f"renders['{k}'] shape {arr.shape} does not match centre shape "
                f"{centre.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"renders['{k}'] must contain only finite values.")
    return centre.shape

def compute_fisher_information_se3(
    renders: dict[str, np.ndarray],
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
    rotation_step_rad: float,
) -> np.ndarray:
    """
    Build the 6x6 SE(3) Fisher information matrix for a composite particle.

    The state vector is ``[x, y, z, omega_x, omega_y, omega_z]`` where the
    omegas are infinitesimal body-fixed rotations about the principal axes
    in the same fixed convention used to render the perturbed images.

    Parameters
    ----------
    renders : dict[str, ndarray]
        Per-pose per-particle contrast images, keyed by:
            ``centre``                — reference pose; used for dC/dx, dC/dy
                                        via central differences in pixel space.
            ``z_minus``, ``z_plus``    — particle translated by ``∓z_step_nm``
                                        in z; used for dC/dz.
            ``rx_minus``, ``rx_plus``  — composite rotated by ``∓rotation_step_rad``
                                        about body-fixed x; used for dC/dω_x.
            ``ry_minus``, ``ry_plus``  — analog about body-fixed y.
            ``rz_minus``, ``rz_plus``  — analog about body-fixed z.
        All must be 2D arrays of identical shape.
    noise_variance_map : 2D array or scalar float
        Pixel-wise variance of the observed image at the centre pose.
    pixel_size_nm : float
        Detector pixel pitch in nanometres. Lateral gradients are reported
        in units of ``[contrast] / nm`` and converted to coordinate
        derivatives under the stationary-shift convention.
    z_step_nm : float
        Axial translation step used to render ``z_minus``/``z_plus``;
        ``dC/dz = (renders['z_plus'] - renders['z_minus']) / (2 * z_step_nm)``.
    rotation_step_rad : float
        Rotation step (radians) used to render the six rotation perturbations;
        ``dC/dω_i = (renders['ri_plus'] - renders['ri_minus']) / (2 * rotation_step_rad)``.

    Returns
    -------
    F : (6, 6) array
        Symmetric Fisher information matrix with state ordering
        ``[x, y, z, omega_x, omega_y, omega_z]``. Translation entries have
        units of ``1 / nm^2``; rotation entries have units of ``1 / rad^2``;
        the off-diagonal coupling entries have units of ``1 / (nm * rad)``.
        Mixed units across the 6x6 matrix are *correct* — the inverse
        produces sigma_x in nm and sigma_omega in rad, both physical.

    Notes
    -----
    The decision to take in-plane gradients from the centre image only
    (rather than from a (-x, +x) and (-y, +y) translation pair) is
    deliberate and matches the 3D variant: under the standard stationary-PSF
    assumption ``I(r; x_0, y_0) = C(r - r_0)``, the coordinate derivatives
    are the negative centre-image spatial gradients. Re-rendering the
    composite at translated x, y would give a numerically equivalent estimate
    at strictly higher cost.
    """
    from .lateral import _build_symmetric_fisher_from_gradients, _lateral_coordinate_derivatives

    _validate_se3_renders(
        renders,
        expected_shape=None,
    )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")
    if not np.isfinite(rotation_step_rad) or rotation_step_rad <= 0.0:
        raise ValueError(
            f"rotation_step_rad must be positive; got {rotation_step_rad}."
        )

    centre = np.asarray(renders["centre"], dtype=float)
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(centre, pixel_size_nm)
    dC_dz = (np.asarray(renders["z_plus"], dtype=float)
             - np.asarray(renders["z_minus"], dtype=float)) / (2.0 * z_step_nm)
    dC_dwx = (np.asarray(renders["rx_plus"], dtype=float)
              - np.asarray(renders["rx_minus"], dtype=float)) / (2.0 * rotation_step_rad)
    dC_dwy = (np.asarray(renders["ry_plus"], dtype=float)
              - np.asarray(renders["ry_minus"], dtype=float)) / (2.0 * rotation_step_rad)
    dC_dwz = (np.asarray(renders["rz_plus"], dtype=float)
              - np.asarray(renders["rz_minus"], dtype=float)) / (2.0 * rotation_step_rad)

    grads = (dI_dx0, dI_dy0, dC_dz, dC_dwx, dC_dwy, dC_dwz)

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
        F = _build_symmetric_fisher_from_gradients(grads, inv_var)
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != centre.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match centre "
                f"render shape {centre.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
        F = _build_symmetric_fisher_from_gradients(grads, inv_var)
    return F

def compute_localization_orientation_crlb(
    renders: dict[str, np.ndarray],
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
    rotation_step_rad: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """
    Joint translation + orientation Cramér-Rao bound for a composite particle.

    Returns
    -------
    result : dict with keys
        - ``sigma_x_nm``, ``sigma_y_nm``, ``sigma_z_nm`` : translation CRLB
          on each axis, in nanometres.
        - ``sigma_xyz_nm``                              : sqrt of summed
          translation variances.
        - ``sigma_omega_x_rad``, ``sigma_omega_y_rad``,
          ``sigma_omega_z_rad``                         : orientation CRLB
          on each body-fixed axis, in radians.
        - ``sigma_omega_total_rad``                     : sqrt of summed
          orientation variances over the *observable* rotation axes (i.e.
          axes with finite per-axis CRLB). For fully orientation-observable
          particles this is the standard isotropic aggregate orientation-
          precision summary; for partial-rank
          composites (e.g. a body-axis-symmetric dimer with omega_x
          singular) this is the joint precision over the observable
          rotation subspace, with the unobservable axes reported separately
          in ``axes_singular``. Returns +inf only when every rotation axis
          is singular (e.g. a perfect sphere). This is *not* a geodesic on
          SO(3); for small angles the small-angle Lie-algebra norm and the
          geodesic norm coincide to first order, which is the regime in
          which a finite-difference Fisher matrix is meaningful.
        - ``fisher_matrix``                             : the 6x6 matrix.
        - ``fisher_det``                                : determinant.
        - ``singular``                                  : True if the full
          6x6 was singular and *some* coordinates' bounds are +inf.
        - ``rank``                                      : numerical rank of
          the estimable state-axis support after singular axes are removed.
        - ``numerical_fisher_rank``                     : raw eigenvalue-based
          rank of the symmetrized Fisher matrix under the degradation
          criterion before axis-estimability clipping.
        - ``axes_singular``                             : list of state-axis
          names whose bound is +inf (e.g. ``['omega_z']`` for an
          axially-symmetric particle whose z-rotation is unobservable).

    Graceful degradation
    --------------------
    Composite particles with axial symmetry have an unobservable rotation
    DOF: the z-axis rotation of a perfect sphere or a body-axis rotation
    of an axisymmetric rod produces zero contrast change, so the
    corresponding Fisher row/column is zero. In that case the routine
    inverts the lower-rank sub-block alone and reports the unobservable
    coordinate's bound as +inf, which is the correct estimator-theoretic
    statement (no unbiased estimator can pin a state variable to which
    the data does not respond).
    """
    F = compute_fisher_information_se3(
        renders, noise_variance_map, pixel_size_nm, z_step_nm, rotation_step_rad,
    )
    derivative_metadata = _se3_derivative_metadata(
        pixel_size_nm,
        z_step_nm,
        rotation_step_rad,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )

    state_axes = ("x", "y", "z", "omega_x", "omega_y", "omega_z")
    sigma_units = ("nm", "nm", "nm", "rad", "rad", "rad")
    F_sym = 0.5 * (F + F.T)
    rank_metadata_all = _fisher_rank_metadata(F_sym)
    if np.any(~np.isfinite(F)):
        axes_singular = list(state_axes)
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_z_nm": float("inf"),
            "sigma_xyz_nm": float("inf"),
            "sigma_omega_x_rad": float("inf"),
            "sigma_omega_y_rad": float("inf"),
            "sigma_omega_z_rad": float("inf"),
            "sigma_omega_total_rad": float("inf"),
            "fisher_matrix": F_sym,
            "fisher_det": float("nan"),
            "singular": True,
            "rank": 0,
            "numerical_fisher_rank": rank_metadata_all["numerical_fisher_rank"],
            "fisher_eigenvalues": rank_metadata_all["fisher_eigenvalues"],
            "fisher_rank_tolerance": rank_metadata_all["fisher_rank_tolerance"],
            "condition_number": rank_metadata_all["condition_number"],
            "axes_singular": axes_singular,
            "state_axes": state_axes,
            "sigma_units": sigma_units,
            "derivative_metadata": derivative_metadata,
            **_diagnostic_metadata_aliases(
                derivative_metadata,
                rank_metadata_all,
                axes_singular=axes_singular,
                sigma_units=sigma_units,
            ),
        }

    diag = np.diag(F_sym)
    trans_scale = max(float(np.max(np.abs(diag[:3]))), 0.0)
    rot_scale = max(float(np.max(np.abs(diag[3:]))), 0.0)
    axis_scales = np.asarray([trans_scale, trans_scale, trans_scale, rot_scale, rot_scale, rot_scale])
    axis_tols = np.maximum(_FISHER_SE3_EPS, axis_scales * _FISHER_SE3_AXIS_RELATIVE_TOL)
    axis_observable = diag > axis_tols
    F_rank = np.array(F_sym, copy=True)
    for axis_index, observable in enumerate(axis_observable):
        if not bool(observable):
            F_rank[axis_index, :] = 0.0
            F_rank[:, axis_index] = 0.0

    try:
        evals, evecs = np.linalg.eigh(F_rank)
    except np.linalg.LinAlgError:
        evals = np.asarray([], dtype=float)
        evecs = np.empty((6, 0), dtype=float)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 0.0)
    rank_tol = max(_FISHER_SE3_EPS, scale * _FISHER_RANK_RELATIVE_TOL)
    positive = evals > rank_tol if evals.size else np.zeros(0, dtype=bool)
    fisher_rank = int(np.count_nonzero(positive))

    sigmas = [float("inf")] * 6
    axes_singular: list[str] = []
    if fisher_rank > 0:
        V = evecs[:, positive]
        inv_evals = 1.0 / evals[positive]
        F_pinv = (V * inv_evals) @ V.T
        range_projector = V @ V.T
        eye = np.eye(6)
        for i, axis in enumerate(state_axes):
            axis_residual = eye[:, i] - range_projector @ eye[:, i]
            axis_estimable = bool(axis_observable[i]) and np.linalg.norm(axis_residual) <= _FISHER_RANGE_RESIDUAL_TOL
            if axis_estimable:
                v = float(F_pinv[i, i])
                sigmas[i] = float(np.sqrt(max(v, 0.0)))
            else:
                axes_singular.append(axis)
    else:
        axes_singular = list(state_axes)

    sigma_x, sigma_y, sigma_z, sigma_wx, sigma_wy, sigma_wz = sigmas

    # Combined translation and orientation summaries. The orientation
    # aggregate is taken over the *observable* rotation axes only (those with
    # finite per-axis CRLB); it returns +inf only when every rotation axis is
    # singular. This makes partial-rank composites (e.g.\ a body-axis-symmetric
    # dimer with omega_x unobservable) rankable on their observable rotation
    # subspace, instead of all collapsing to +inf and becoming
    # cross-modality-indistinguishable. The translation aggregate keeps the
    # strict "all 3 finite" semantics because position observability is the
    # generic case for any imaging modality.
    finite_trans = [s for s in (sigma_x, sigma_y, sigma_z) if np.isfinite(s)]
    finite_rot = [s for s in (sigma_wx, sigma_wy, sigma_wz) if np.isfinite(s)]
    sigma_xyz = (
        float(np.sqrt(sum(s * s for s in finite_trans)))
        if len(finite_trans) == 3 else float("inf")
    )
    sigma_omega_total = (
        float(np.sqrt(sum(s * s for s in finite_rot)))
        if len(finite_rot) > 0 else float("inf")
    )
    observable_axis_rank = int(len(state_axes) - len(set(axes_singular)))
    reported_rank = int(min(fisher_rank, observable_axis_rank))

    return {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_z_nm": sigma_z,
        "sigma_xyz_nm": sigma_xyz,
        "sigma_omega_x_rad": sigma_wx,
        "sigma_omega_y_rad": sigma_wy,
        "sigma_omega_z_rad": sigma_wz,
        "sigma_omega_total_rad": sigma_omega_total,
        "fisher_matrix": F_sym,
        "fisher_det": float(np.linalg.det(F_sym)),
        "singular": reported_rank < 6,
        "rank": reported_rank,
        "numerical_fisher_rank": fisher_rank,
        "fisher_eigenvalues": evals.astype(float).tolist(),
        "fisher_rank_tolerance": float(rank_tol),
        "condition_number": (
            float(np.max(evals[positive]) / np.min(evals[positive]))
            if np.any(positive) and float(np.min(evals[positive])) > 0.0
            else float("inf")
        ),
        "axis_observable_tolerances": axis_tols.astype(float).tolist(),
        "axis_observable": [bool(flag) for flag in axis_observable],
        "axes_singular": axes_singular,
        "state_axes": state_axes,
        "sigma_units": sigma_units,
        "derivative_metadata": derivative_metadata,
        **_diagnostic_metadata_aliases(
            derivative_metadata,
            {
                "fisher_eigenvalues": evals.astype(float).tolist(),
                "fisher_rank_tolerance": float(rank_tol),
                "numerical_fisher_rank": fisher_rank,
                "condition_number": (
                    float(np.max(evals[positive]) / np.min(evals[positive]))
                    if np.any(positive) and float(np.min(evals[positive])) > 0.0
                    else float("inf")
                ),
            },
            axes_singular=axes_singular,
            sigma_units=sigma_units,
        ),
    }

def _validate_rank_int(value: Any, *, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]; got {value!r}.")
    value_int = int(value)
    if value_int < lower or value_int > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]; got {value_int}.")
    return value_int

def predict_se3_rank_from_symmetry(
    continuous_rotational_symmetry_dim: int,
    translation_rank: int = 3,
    *,
    rotational_dimension: int = 3,
    singular_rotation_axes_body: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Predict local SE(3) Fisher rank from continuous rotational symmetry.

    A connected stabilizer subgroup of dimension d contributes at least d null
    rotation directions. Under the generic local-observability condition, the
    rotational Fisher rank is ``rotational_dimension - d`` and the full SE(3)
    rank is ``translation_rank + rotational_rank``. Discrete symmetries are not
    counted here because they create global pose aliases, not infinitesimal null
    directions.
    """
    rotational_dimension = _validate_rank_int(
        rotational_dimension,
        name="rotational_dimension",
        lower=1,
        upper=3,
    )
    symmetry_dim = _validate_rank_int(
        continuous_rotational_symmetry_dim,
        name="continuous_rotational_symmetry_dim",
        lower=0,
        upper=rotational_dimension,
    )
    translation_rank = _validate_rank_int(
        translation_rank,
        name="translation_rank",
        lower=0,
        upper=3,
    )
    axes = list(singular_rotation_axes_body or [])
    allowed_axes = set(SE3_STATE_AXES)
    invalid_axes = [axis for axis in axes if axis not in allowed_axes]
    if invalid_axes:
        raise ValueError(
            "singular_rotation_axes_body contains unsupported axis names: "
            f"{invalid_axes!r}."
        )

    rotational_rank = rotational_dimension - symmetry_dim
    se3_rank = translation_rank + rotational_rank
    predicted_nullity = (3 - translation_rank) + symmetry_dim

    return {
        "continuous_rotational_symmetry_dim": symmetry_dim,
        "translation_rank": translation_rank,
        "rotational_dimension": rotational_dimension,
        "predicted_rotational_rank": int(rotational_rank),
        "predicted_se3_rank": int(se3_rank),
        "predicted_nullity": int(predicted_nullity),
        "symmetry_nullity_lower_bound": int(symmetry_dim),
        "singular_rotation_axes_body": axes,
    }

def predict_fused_se3_rank_from_symmetry(
    continuous_rotational_symmetry_intersection_dim: int,
    *,
    per_modality_symmetry_dims: dict[str, int] | None = None,
    translation_rank: int = 3,
    rotational_dimension: int = 3,
) -> dict[str, Any]:
    """
    Predict fused SE(3) Fisher rank from the intersection stabilizer dimension.

    For independent modalities on the same SE(3) parameter frame, the fused
    Fisher derivative along a rotation generator is zero iff every contributing
    modality is invariant along that generator. The continuous stabilizer of the
    fused contrast vector is therefore the intersection of the per-modality
    continuous stabilizers. This helper takes that intersection dimension
    explicitly; dimensions of individual stabilizers alone are not enough to
    infer the intersection.
    """
    prediction = predict_se3_rank_from_symmetry(
        continuous_rotational_symmetry_intersection_dim,
        translation_rank=translation_rank,
        rotational_dimension=rotational_dimension,
    )
    modality_dims: dict[str, int] = {}
    if per_modality_symmetry_dims is not None:
        for modality, dim in per_modality_symmetry_dims.items():
            modality_dims[str(modality)] = _validate_rank_int(
                dim,
                name=f"per_modality_symmetry_dims[{modality!r}]",
                lower=0,
                upper=rotational_dimension,
            )
    intersection_dim = prediction["continuous_rotational_symmetry_dim"]
    symmetry_broken = (
        any(dim > intersection_dim for dim in modality_dims.values())
        if modality_dims else None
    )
    return {
        **prediction,
        "continuous_rotational_symmetry_intersection_dim": intersection_dim,
        "per_modality_continuous_rotational_symmetry_dim": modality_dims,
        "symmetry_broken_by_fusion": symmetry_broken,
        "fusion_rank_prediction_note": (
            "Fusion nullity is set by the intersection of contrast-functional "
            "continuous stabilizers; per-modality stabilizer dimensions alone "
            "do not determine that intersection."
        ),
    }

def compare_observed_and_predicted_se3_rank(
    crlb_result: dict[str, Any],
    symmetry_metadata: dict[str, Any] | None,
    *,
    translation_rank: int = 3,
) -> dict[str, Any]:
    """
    Compare an observed SE(3) CRLB rank against the symmetry-rank prediction.

    Missing symmetry metadata is reported explicitly instead of guessed.
    """
    observed_axis_rank_raw = crlb_result.get("rank", None)
    observed_axis_rank = (
        None if observed_axis_rank_raw is None else _validate_rank_int(
            observed_axis_rank_raw,
            name="crlb_result['rank']",
            lower=0,
            upper=6,
        )
    )
    observed_matrix_rank_raw = crlb_result.get(
        "numerical_fisher_rank",
        observed_axis_rank_raw,
    )
    observed_matrix_rank = (
        None if observed_matrix_rank_raw is None else _validate_rank_int(
            observed_matrix_rank_raw,
            name="crlb_result['numerical_fisher_rank']",
            lower=0,
            upper=6,
        )
    )
    observed_axes = list(crlb_result.get("axes_singular", []))
    if not symmetry_metadata or symmetry_metadata.get("continuous_rotational_symmetry_dim") is None:
        return {
            "rank_prediction_available": False,
            "observed_rank": observed_matrix_rank,
            "observed_matrix_rank": observed_matrix_rank,
            "observed_axis_estimability_rank": observed_axis_rank,
            "observed_axes_singular": observed_axes,
            "rank_matches_symmetry_prediction": None,
            "satisfies_symmetry_nullity_bound": None,
            "rank_prediction_note": "No continuous_rotational_symmetry_dim metadata was supplied.",
        }

    prediction = predict_se3_rank_from_symmetry(
        symmetry_metadata["continuous_rotational_symmetry_dim"],
        translation_rank=translation_rank,
        singular_rotation_axes_body=symmetry_metadata.get("singular_rotation_axes_body"),
    )
    if observed_matrix_rank is None:
        note = "Observed CRLB result did not contain a rank."
        matches = None
        satisfies = None
        observed_nullity = None
    else:
        observed_nullity = 6 - observed_matrix_rank
        matches = observed_matrix_rank == prediction["predicted_se3_rank"]
        satisfies = observed_nullity >= prediction["symmetry_nullity_lower_bound"]
        if matches:
            note = "Observed numerical Fisher rank matches the generic symmetry prediction."
        elif satisfies:
            note = (
                "Observed numerical Fisher rank satisfies the symmetry nullity lower bound but "
                "shows additional degeneracy beyond the generic prediction."
            )
        else:
            note = (
                "Observed numerical Fisher rank violates the symmetry nullity lower bound; check "
                "symmetry metadata, render convention, or rank tolerance."
            )

    return {
        **prediction,
        "rank_prediction_available": True,
        "observed_rank": observed_matrix_rank,
        "observed_matrix_rank": observed_matrix_rank,
        "observed_axis_estimability_rank": observed_axis_rank,
        "observed_axes_singular": observed_axes,
        "observed_nullity": observed_nullity,
        "rank_matches_symmetry_prediction": matches,
        "satisfies_symmetry_nullity_bound": satisfies,
        "rank_prediction_note": note,
    }

def compare_modality_orientation_crlb(
    renders_by_modality: dict[str, dict[str, np.ndarray]],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float,
    rotation_step_rad: float,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    measurement_domain_by_modality: str | dict[str, str] | None = None,
    signal_units_by_modality: str | dict[str, str] | None = None,
    noise_variance_units_by_modality: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Order imaging modalities by the orientation Cramér-Rao bound they deliver
    on a shared composite-particle configuration.

    For each modality ``M`` the caller supplies a dict of nine perturbed
    renders (``centre``, ``z_minus/plus``, ``rx_minus/plus``,
    ``ry_minus/plus``, ``rz_minus/plus``) of the *same* composite particle
    plus a noise variance map. Each modality's renders are passed through
    :func:`compute_localization_orientation_crlb` to obtain the per-axis
    sigmas and the aggregate ``sigma_omega_total_rad``. The modalities are
    then ordered by ``sigma_omega_total_rad``, smallest first. A modality with
    only some observable rotation axes receives a finite aggregate over those
    axes plus explicit ``axes_singular`` metadata; a modality with no observable
    orientation axes reports ``sigma_omega_total_rad = +inf`` and sorts to the
    end of the ordering.

    Parameters
    ----------
    renders_by_modality : dict[str, dict[str, np.ndarray]]
        Outer key = modality name; inner dict = the nine perturbed renders
        accepted by :func:`compute_localization_orientation_crlb`. The
        rendered images must have the same ``centre`` shape across modalities
        --- they describe the *same* particle under different contrast
        mechanisms; pixel pitch and noise floor scale are common.
    noise_variance_by_modality : dict[str, ndarray or float]
        Per-modality noise-variance map (or scalar). Keys must match
        ``renders_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch (nm). A scalar keeps the historical shared-pitch
        behavior; a mapping supplies one pitch per modality.
    z_step_nm : float
        Axial translation step used to render ``z_minus``/``z_plus`` for
        every modality. Must match the step used to actually render.
    rotation_step_rad : float
        Body-frame rotation step (radians) used to render the six rotation
        perturbations for every modality.

    Returns
    -------
    dict with keys
        - ``per_modality``           : dict[str, dict] from
                                       :func:`compute_localization_orientation_crlb`,
                                       one entry per modality.
        - ``ordering``               : list of ``(modality, sigma_omega_total_rad)``
                                       sorted ascending; +inf entries last.
                                       ``ranking`` is retained as an alias.
        - ``best_modality``          : argmin ``sigma_omega_total_rad`` over
                                       the modalities for which the orientation
                                       block has at least one observable axis;
                                       ``None`` if no modality recovers any
                                       rotation axis.
        - ``best_modality_observable`` : synonym for ``best_modality``.
        - ``best_modality_full_rank``  : argmin among finite modalities whose
                                       full six-parameter block is observable;
                                       ``None`` if no modality is full rank.
        - ``relative_sigma_omega``   : dict[modality -> sigma / sigma_best];
                                       +inf when sigma is +inf or no best.
        - ``frames_to_match_best``   : dict[modality -> rho^2]; the equivalent-
                                       frame budget required for modality M to
                                       match one frame of the lowest-bound
                                       modality on *orientation* precision.
        - ``axes_singular_per_modality`` : dict[modality -> list[str]],
                                       per-axis observability flags.

    Notes
    -----
    The orientation comparator differs from
    :func:`compare_modality_information_content` in that the relevant
    summary statistic is ``sigma_omega_total_rad`` (an aggregate over the
    three rotation axes in radians) rather than ``sigma_xy_nm``. The
    equivalent-frame-budget formula carries through unchanged because the
    Fisher information is additive across independent frames for *all*
    state coordinates --- including rotational ones --- under the same
    Gaussian-noise approximation used throughout.
    """
    from .lateral import _resolve_modality_scalar_map, _resolve_modality_string_map, _sort_key_finite_then_value

    if not isinstance(renders_by_modality, dict) or not renders_by_modality:
        raise ValueError(
            "renders_by_modality must be a non-empty dict keyed by modality name."
        )
    if not isinstance(noise_variance_by_modality, dict):
        raise ValueError("noise_variance_by_modality must be a dict.")
    if set(renders_by_modality) != set(noise_variance_by_modality):
        missing = set(renders_by_modality) - set(noise_variance_by_modality)
        extra = set(noise_variance_by_modality) - set(renders_by_modality)
        raise ValueError(
            "renders_by_modality and noise_variance_by_modality keys must match; "
            f"missing noise entries: {sorted(missing)}; extra noise entries: {sorted(extra)}."
        )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")
    if not np.isfinite(rotation_step_rad) or rotation_step_rad <= 0.0:
        raise ValueError(
            f"rotation_step_rad must be positive; got {rotation_step_rad}."
        )

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        renders_by_modality.keys(),
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )
    measurement_domains = _resolve_modality_string_map(
        measurement_domain_by_modality,
        renders_by_modality.keys(),
        "contrast",
    )
    signal_units = _resolve_modality_string_map(
        signal_units_by_modality,
        renders_by_modality.keys(),
        "contrast",
    )
    noise_variance_units = _resolve_modality_string_map(
        noise_variance_units_by_modality,
        renders_by_modality.keys(),
        "",
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, renders in renders_by_modality.items():
        try:
            per_modality[modality] = compute_localization_orientation_crlb(
                renders=renders,
                noise_variance_map=noise_variance_by_modality[modality],
                pixel_size_nm=pixel_sizes[modality],
                z_step_nm=z_step_nm,
                rotation_step_rad=rotation_step_rad,
                signal_units=signal_units[modality],
                measurement_domain=measurement_domains[modality],
                noise_variance_units=(
                    noise_variance_units[modality] or None
                ),
            )
        except Exception as exc:  # noqa: BLE001 — record per-modality failure without aborting comparison
            per_modality[modality] = {
                "error": repr(exc),
                "sigma_x_nm": float("inf"),
                "sigma_y_nm": float("inf"),
                "sigma_z_nm": float("inf"),
                "sigma_xyz_nm": float("inf"),
                "sigma_omega_x_rad": float("inf"),
                "sigma_omega_y_rad": float("inf"),
                "sigma_omega_z_rad": float("inf"),
                "sigma_omega_total_rad": float("inf"),
                "fisher_matrix": None,
                "fisher_det": None,
                "singular": True,
                "rank": 0,
                "axes_singular": ["x", "y", "z", "omega_x", "omega_y", "omega_z"],
                "state_axes": ("x", "y", "z", "omega_x", "omega_y", "omega_z"),
                "sigma_units": ("nm", "nm", "nm", "rad", "rad", "rad"),
            }

    # Build the (modality, sigma_omega_total) tuples and sort.
    items = [
        (m, float(r.get("sigma_omega_total_rad", float("inf"))))
        for m, r in per_modality.items()
    ]
    # Sort ascending; +inf and NaN go last.
    ranking = sorted(items, key=_sort_key_finite_then_value)

    # ``best_modality`` follows the documented observable-axis contract. The
    # stricter full-rank winner is exposed separately for callers that need all
    # six localization/orientation coordinates to be jointly observable.
    best_modality: str | None = None
    best_modality_full_rank: str | None = None
    for modality, sigma in ranking:
        rec = per_modality[modality]
        if best_modality is None and np.isfinite(sigma):
            best_modality = modality
        if (
            best_modality_full_rank is None
            and np.isfinite(sigma)
            and rec.get("rank", 0) == 6
            and not rec.get("axes_singular", [])
        ):
            best_modality_full_rank = modality
        if best_modality is not None and best_modality_full_rank is not None:
            break

    # Relative-precision and equivalent-frame-budget against best.
    if best_modality is None:
        relative = init_infinite_dict(per_modality)
        frames = init_infinite_dict(per_modality)
    else:
        sigma_best = float(per_modality[best_modality]["sigma_omega_total_rad"])
        relative = {}
        frames = {}
        for m, s in items:
            if not np.isfinite(s) or sigma_best <= 0.0:
                relative[m] = float("inf")
                frames[m] = float("inf")
            else:
                rho = s / sigma_best
                relative[m] = float(rho)
                frames[m] = float(rho * rho)

    axes_singular_per_modality = {
        m: list(per_modality[m].get("axes_singular", [])) for m in per_modality
    }

    return {
        "per_modality": per_modality,
        "ordering": ranking,
        "ranking": ranking,
        "best_modality": best_modality,
        "best_modality_observable": best_modality,
        "best_modality_full_rank": best_modality_full_rank,
        "relative_sigma_omega": relative,
        "frames_to_match_best": frames,
        "axes_singular_per_modality": axes_singular_per_modality,
        "pixel_size_nm_by_modality": pixel_sizes,
        "measurement_domain_by_modality": measurement_domains,
        "signal_units_by_modality": signal_units,
        "noise_variance_units_by_modality": {
            modality: (
                noise_variance_units[modality]
                or _variance_units(signal_units[modality])
            )
            for modality in renders_by_modality
        },
    }

__all__ = ['compute_fisher_information_se3', 'compute_localization_orientation_crlb', 'predict_se3_rank_from_symmetry', 'predict_fused_se3_rank_from_symmetry', 'compare_observed_and_predicted_se3_rank', 'compare_modality_orientation_crlb']
