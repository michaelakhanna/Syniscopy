"""Physical Monte Carlo SEM transport backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.runtime import SemSettings, param_value, resolved_sem_monte_carlo_seed
from material_optical_catalog import SEMTransportMaterial
from imaging_models.sem_source import SEMMaterialChannelKey, SEMMaterialSourceCanvas

from ._metadata import (
    SEMTransportBackendError,
    SEMTransportMetadata,
    _detector_takeoff_acceptance_gain,
    _electrons_from_beam_current,
    _fft_convolve_centered,
    _finite_nonnegative,
    _gradient_components,
    _gradient_magnitude,
    attach_backend_fidelity_metadata,
    np,
)


_AVOGADRO_PER_MOL = 6.02214076e23
_KEV_CUTOFF_DEFAULT = 0.05
_CM_TO_NM = 1.0e7
_SEM_PHYSICAL_ELASTIC_MODELS = {
    "screened_rutherford",
    "mott_browning",
}


@dataclass(frozen=True)
class SEMTransportObservables:
    backscatter_coefficient: float
    max_penetration_depth_nm: float
    p99_penetration_depth_nm: float
    active_histories_at_limit: int
    histories: int
    steps_executed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backscatter_coefficient": self.backscatter_coefficient,
            "max_penetration_depth_nm": self.max_penetration_depth_nm,
            "p99_penetration_depth_nm": self.p99_penetration_depth_nm,
            "active_histories_at_limit": self.active_histories_at_limit,
            "histories": self.histories,
            "steps_executed": self.steps_executed,
        }


@dataclass(frozen=True)
class SEMPhysicalKernelBundle:
    material_key: SEMMaterialChannelKey
    observables: SEMTransportObservables
    kernel: np.ndarray
    kernel_stack: np.ndarray | None
    raw_surface_energy_tally: float
    raw_volume_energy_tally: float
    surface_escape_energy_fraction_per_primary: float
    volume_escape_energy_fraction_per_primary: float
    kernel_size_px: int


def _radial_average_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return the rotational expectation of a normal-incidence SEM kernel."""
    arr = np.asarray(kernel, dtype=float)
    if arr.ndim != 2:
        raise SEMTransportBackendError(f"SEM interaction kernel must be 2D; got shape {arr.shape!r}.")
    if arr.shape[0] != arr.shape[1] or arr.shape[0] % 2 == 0:
        raise SEMTransportBackendError(f"SEM interaction kernel must be odd and square; got shape {arr.shape!r}.")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("SEM interaction kernel contains non-finite values.")
    center = arr.shape[0] // 2
    yy, xx = np.indices(arr.shape, dtype=float)
    radial_bin = np.rint(np.sqrt((xx - center) ** 2 + (yy - center) ** 2)).astype(int)
    sums = np.bincount(radial_bin.ravel(), weights=arr.ravel())
    counts = np.bincount(radial_bin.ravel())
    profile = sums / np.maximum(counts, 1)
    averaged = profile[radial_bin]
    total = float(np.sum(averaged))
    if total <= 0.0 or not np.isfinite(total):
        raise SEMTransportBackendError("Radially averaged SEM interaction kernel has no finite energy tally.")
    return averaged / total


def _radial_average_kernel_stack(kernel_stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(kernel_stack, dtype=float)
    if stack.ndim != 3:
        raise SEMTransportBackendError(f"SEM volume interaction kernel must be 3D; got shape {stack.shape!r}.")
    averaged = np.stack([_radial_average_kernel(slice_kernel) for slice_kernel in stack], axis=0)
    slice_weights = np.sum(stack, axis=(1, 2))
    if not np.all(np.isfinite(slice_weights)) or float(np.sum(slice_weights)) <= 0.0:
        raise SEMTransportBackendError("SEM volume interaction kernel has no finite depth energy tally.")
    averaged *= slice_weights[:, None, None]
    total = float(np.sum(averaged))
    if total <= 0.0 or not np.isfinite(total):
        raise SEMTransportBackendError("Radially averaged SEM volume interaction kernel has no finite energy tally.")
    return averaged / total


def sem_transport_material_from_channel_key(key: SEMMaterialChannelKey) -> SEMTransportMaterial:
    return SEMTransportMaterial(
        name=key.material_name,
        atomic_number=key.atomic_number,
        atomic_weight_g_mol=key.atomic_weight_g_mol,
        density_g_cm3=key.density_g_cm3,
        se_yield_coefficient=key.se_yield_coefficient,
    )


def normalize_sem_physical_elastic_model(raw: object) -> str:
    model = str(raw).strip().lower()
    if model not in _SEM_PHYSICAL_ELASTIC_MODELS:
        raise SEMTransportBackendError(
            "PARAMS['sem_physical_elastic_model'] must be one of "
            f"{sorted(_SEM_PHYSICAL_ELASTIC_MODELS)}; got {raw!r}."
        )
    return model


def _rutherford_screening_alpha(
    energy_keV: np.ndarray | float,
    atomic_number: float,
) -> np.ndarray:
    energy = np.maximum(np.asarray(energy_keV, dtype=float), 1e-12)
    return 3.4e-3 * (float(atomic_number) ** 0.67) / energy


def screened_rutherford_cross_section_cm2(
    energy_keV: np.ndarray | float,
    atomic_number: float,
) -> np.ndarray:
    """Total screened-Rutherford elastic cross-section per atom in cm^2."""
    energy = np.maximum(np.asarray(energy_keV, dtype=float), 1e-12)
    z = float(atomic_number)
    alpha = _rutherford_screening_alpha(energy, z)
    relativistic = ((energy + 511.0) / (energy + 1022.0)) ** 2
    return (
        5.21e-21
        * ((z * z) / (energy * energy))
        * (4.0 * np.pi / (alpha * (1.0 + alpha)))
        * relativistic
    )


def browning_mott_cross_section_cm2(
    energy_keV: np.ndarray | float,
    atomic_number: float,
) -> np.ndarray:
    """Browning empirical total elastic cross-section fitted to Mott data.

    Energy is in keV and the returned cross-section is cm^2 per atom. This is a
    fast analytic Mott surrogate, not a replacement for NIST SRD/Czyzewski
    tabulated partial-wave cross-sections.
    """
    energy = np.maximum(np.asarray(energy_keV, dtype=float), 1e-12)
    z = float(atomic_number)
    sqrt_energy = np.sqrt(energy)
    denominator = (
        energy
        + 0.005 * (z ** 1.7) * sqrt_energy
        + 0.0007 * (z * z) / np.maximum(sqrt_energy, 1e-12)
    )
    return 3.0e-18 * (z ** 1.7) / np.maximum(denominator, 1e-30)


def elastic_cross_section_cm2(
    energy_keV: np.ndarray | float,
    material: SEMTransportMaterial,
    *,
    elastic_model: str,
) -> np.ndarray:
    model = normalize_sem_physical_elastic_model(elastic_model)
    if model == "screened_rutherford":
        return screened_rutherford_cross_section_cm2(energy_keV, material.atomic_number)
    if model == "mott_browning":
        return browning_mott_cross_section_cm2(energy_keV, material.atomic_number)
    raise AssertionError(model)


def elastic_mean_free_path_nm(
    energy_keV: np.ndarray | float,
    material: SEMTransportMaterial,
    *,
    elastic_model: str,
) -> np.ndarray:
    sigma = elastic_cross_section_cm2(
        energy_keV,
        material,
        elastic_model=elastic_model,
    )
    lambda_cm = material.atomic_weight_g_mol / (
        _AVOGADRO_PER_MOL * material.density_g_cm3 * sigma
    )
    return lambda_cm * _CM_TO_NM


def joy_luo_stopping_power_keV_per_nm(
    energy_keV: np.ndarray | float,
    material: SEMTransportMaterial,
) -> np.ndarray:
    energy = np.maximum(np.asarray(energy_keV, dtype=float), 1e-12)
    z = float(material.atomic_number)
    j_keV = (9.76 * z + 58.5 * (z ** -0.19)) * 1.0e-3
    log_arg = 1.166 * (energy + 0.85 * j_keV) / j_keV
    stopping_keV_per_cm = (
        7.85e4
        * (material.density_g_cm3 * z)
        / (material.atomic_weight_g_mol * energy)
        * np.log(np.maximum(log_arg, 1.0 + 1e-12))
    )
    return stopping_keV_per_cm / _CM_TO_NM


def kanaya_okayama_range_nm(energy_keV: float, material: SEMTransportMaterial) -> float:
    range_um = (
        0.0276
        * material.atomic_weight_g_mol
        * (float(energy_keV) ** 1.67)
        / ((material.atomic_number ** 0.889) * material.density_g_cm3)
    )
    return float(range_um * 1000.0)


def reuter_backscatter_coefficient_20kev(atomic_number: float) -> float:
    z = float(atomic_number)
    return float(-0.0254 + 0.016 * z - 1.86e-4 * z * z + 8.3e-7 * z ** 3)


def _screened_rutherford_cos_theta(
    energy_keV: np.ndarray | float,
    atomic_number: float,
    random_u: np.ndarray,
) -> np.ndarray:
    alpha = _rutherford_screening_alpha(energy_keV, atomic_number)
    cos_theta = 1.0 - (2.0 * alpha * random_u) / (1.0 + alpha - random_u)
    return np.clip(cos_theta, -1.0, 1.0)


def _browning_rutherford_to_isotropic_ratio(
    energy_keV: np.ndarray | float,
    atomic_number: float,
) -> np.ndarray:
    energy = np.maximum(np.asarray(energy_keV, dtype=float), 1e-12)
    z = float(atomic_number)
    return (300.0 * energy / z) + ((z ** 3) / (3.0e5 * energy))


def sample_elastic_scattering_cos_theta(
    *,
    rng: np.random.Generator,
    energy_keV: np.ndarray,
    material: SEMTransportMaterial,
    elastic_model: str,
) -> np.ndarray:
    model = normalize_sem_physical_elastic_model(elastic_model)
    random_u = rng.random(energy_keV.size)
    if model == "screened_rutherford":
        return _screened_rutherford_cos_theta(
            energy_keV,
            material.atomic_number,
            random_u,
        )
    if model == "mott_browning":
        ratio = _browning_rutherford_to_isotropic_ratio(
            energy_keV,
            material.atomic_number,
        )
        rutherford_probability = ratio / (1.0 + ratio)
        use_rutherford = rng.random(energy_keV.size) < rutherford_probability
        cos_theta = 2.0 * random_u - 1.0
        if np.any(use_rutherford):
            cos_theta[use_rutherford] = _screened_rutherford_cos_theta(
                energy_keV[use_rutherford],
                material.atomic_number,
                random_u[use_rutherford],
            )
        return np.clip(cos_theta, -1.0, 1.0)
    raise AssertionError(model)


def _rotate_directions(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    cos_theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    denom = np.sqrt(np.maximum(1e-30, 1.0 - uz * uz))
    regular = np.abs(uz) < 0.999999
    nx = np.empty_like(ux)
    ny = np.empty_like(uy)
    nz = np.empty_like(uz)

    nx[regular] = (
        ux[regular] * cos_theta[regular]
        + sin_theta[regular]
        * (
            ux[regular] * uz[regular] * cos_phi[regular]
            - uy[regular] * sin_phi[regular]
        )
        / denom[regular]
    )
    ny[regular] = (
        uy[regular] * cos_theta[regular]
        + sin_theta[regular]
        * (
            uy[regular] * uz[regular] * cos_phi[regular]
            + ux[regular] * sin_phi[regular]
        )
        / denom[regular]
    )
    nz[regular] = (
        uz[regular] * cos_theta[regular]
        - sin_theta[regular] * cos_phi[regular] * denom[regular]
    )

    singular = ~regular
    sign = np.where(uz[singular] >= 0.0, 1.0, -1.0)
    nx[singular] = sin_theta[singular] * cos_phi[singular]
    ny[singular] = sin_theta[singular] * sin_phi[singular]
    nz[singular] = sign * cos_theta[singular]

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / norm, ny / norm, nz / norm


def simulate_sem_transport_observables(
    *,
    material: SEMTransportMaterial,
    acceleration_keV: float,
    histories: int,
    seed: int,
    max_steps: int,
    cutoff_keV: float,
    elastic_model: str,
    probe_sigma_nm: float = 0.0,
    canvas_pitch_nm: float | None = None,
    kernel_size_px: int | None = None,
    escape_depth_nm: float | None = None,
    volume_slices: int | None = None,
    volume_slice_thickness_nm: float | None = None,
) -> tuple[SEMTransportObservables, np.ndarray | None, np.ndarray | None, float, float]:
    """Run seeded vectorized SEM electron histories.

    Returns observables plus optional projected and depth-resolved interaction
    kernels. The kernels tally Joy-Luo energy loss weighted by secondary-electron
    escape probability.
    """
    history_count = int(histories)
    if history_count <= 0:
        raise SEMTransportBackendError("SEM physical MC histories must be positive.")
    step_limit = int(max_steps)
    if step_limit <= 0:
        raise SEMTransportBackendError("SEM physical MC max steps must be positive.")
    energy0 = float(acceleration_keV)
    if not np.isfinite(energy0) or energy0 <= 0.0:
        raise SEMTransportBackendError("SEM physical MC acceleration energy must be positive.")
    cutoff = float(cutoff_keV)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise SEMTransportBackendError("SEM physical MC cutoff energy must be positive.")
    elastic_model = normalize_sem_physical_elastic_model(elastic_model)

    use_kernel = canvas_pitch_nm is not None and kernel_size_px is not None
    if use_kernel:
        pitch = float(canvas_pitch_nm)
        size = int(kernel_size_px)
        if not np.isfinite(pitch) or pitch <= 0.0:
            raise SEMTransportBackendError("SEM physical MC canvas pitch must be positive.")
        if size <= 0:
            raise SEMTransportBackendError("SEM physical MC kernel size must be positive.")
        if size % 2 == 0:
            size += 1
        kernel = np.zeros((size, size), dtype=float)
        center = size // 2
    else:
        pitch = 1.0
        size = 0
        center = 0
        kernel = None

    use_stack = (
        use_kernel
        and volume_slices is not None
        and volume_slice_thickness_nm is not None
    )
    if use_stack:
        slices = int(volume_slices)
        slice_thickness = float(volume_slice_thickness_nm)
        if slices <= 0 or not np.isfinite(slice_thickness) or slice_thickness <= 0.0:
            raise SEMTransportBackendError("SEM physical MC volume kernel geometry is invalid.")
        kernel_stack = np.zeros((slices, size, size), dtype=float)
    else:
        slices = 0
        slice_thickness = 1.0
        kernel_stack = None

    rng = np.random.default_rng(int(seed))
    energy = np.full(history_count, energy0, dtype=float)
    x = rng.normal(0.0, float(probe_sigma_nm), size=history_count) if probe_sigma_nm > 0.0 else np.zeros(history_count)
    y = rng.normal(0.0, float(probe_sigma_nm), size=history_count) if probe_sigma_nm > 0.0 else np.zeros(history_count)
    z = np.zeros(history_count, dtype=float)
    ux = np.zeros(history_count, dtype=float)
    uy = np.zeros(history_count, dtype=float)
    uz = np.ones(history_count, dtype=float)
    max_depth = np.zeros(history_count, dtype=float)
    active = np.ones(history_count, dtype=bool)
    backscattered = np.zeros(history_count, dtype=bool)
    raw_surface_energy_tally = 0.0
    raw_volume_energy_tally = 0.0
    escape_depth = (
        max(float(escape_depth_nm), 1e-12)
        if escape_depth_nm is not None
        else 1.0
    )
    steps_executed = 0

    for step_index in range(step_limit):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        steps_executed = step_index + 1
        energy_before = energy[active_indices]
        lambda_nm = elastic_mean_free_path_nm(
            energy_before,
            material,
            elastic_model=elastic_model,
        )
        sampled_step_nm = -lambda_nm * np.log(np.maximum(rng.random(active_indices.size), 1e-300))
        stopping = joy_luo_stopping_power_keV_per_nm(energy_before, material)
        energy_to_cutoff = np.maximum(energy_before - cutoff, 0.0)
        distance_to_cutoff_nm = np.where(
            stopping > 0.0,
            energy_to_cutoff / np.maximum(stopping, 1e-300),
            np.inf,
        )
        step_nm = np.minimum(sampled_step_nm, distance_to_cutoff_nm)
        old_x = x[active_indices]
        old_y = y[active_indices]
        old_z = z[active_indices]
        new_x = old_x + ux[active_indices] * step_nm
        new_y = old_y + uy[active_indices] * step_nm
        new_z = old_z + uz[active_indices] * step_nm
        deposited_keV = np.minimum(stopping * step_nm, energy_before)
        energy_after = np.maximum(energy_before - deposited_keV, 0.0)

        x[active_indices] = new_x
        y[active_indices] = new_y
        z[active_indices] = new_z
        energy[active_indices] = energy_after
        max_depth[active_indices] = np.maximum(max_depth[active_indices], new_z)

        exited = new_z < 0.0
        if np.any(exited):
            exit_indices = active_indices[exited]
            backscattered[exit_indices] = energy[exit_indices] > cutoff
            active[exit_indices] = False

        stopped = (energy_after <= cutoff) | (sampled_step_nm >= distance_to_cutoff_nm)
        if np.any(stopped):
            active[active_indices[stopped]] = False

        deposit_mask = (~exited) & (deposited_keV > 0.0)
        if np.any(deposit_mask) and use_kernel:
            mid_x = 0.5 * (old_x[deposit_mask] + new_x[deposit_mask])
            mid_y = 0.5 * (old_y[deposit_mask] + new_y[deposit_mask])
            mid_z = np.maximum(0.5 * (old_z[deposit_mask] + new_z[deposit_mask]), 0.0)
            weights = deposited_keV[deposit_mask] * np.exp(-mid_z / escape_depth)
            px = np.floor(center + mid_x / pitch).astype(int)
            py = np.floor(center + mid_y / pitch).astype(int)
            inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            if np.any(inside):
                np.add.at(kernel, (py[inside], px[inside]), weights[inside])
                raw_surface_energy_tally += float(np.sum(weights[inside]))
            if use_stack:
                depth_index = np.floor(mid_z / slice_thickness).astype(int)
                inside_stack = inside & (depth_index >= 0) & (depth_index < slices)
                if np.any(inside_stack):
                    np.add.at(
                        kernel_stack,
                        (depth_index[inside_stack], py[inside_stack], px[inside_stack]),
                        weights[inside_stack],
                    )
                    raw_volume_energy_tally += float(np.sum(weights[inside_stack]))

        continuing = active_indices[(~exited) & (~stopped)]
        if continuing.size == 0:
            continue
        energy_cont = energy[continuing]
        cos_theta = sample_elastic_scattering_cos_theta(
            rng=rng,
            energy_keV=energy_cont,
            material=material,
            elastic_model=elastic_model,
        )
        phi = 2.0 * np.pi * rng.random(continuing.size)
        ux_new, uy_new, uz_new = _rotate_directions(
            ux[continuing],
            uy[continuing],
            uz[continuing],
            cos_theta,
            phi,
        )
        ux[continuing] = ux_new
        uy[continuing] = uy_new
        uz[continuing] = uz_new

    if kernel is not None:
        total = float(np.sum(kernel))
        if total <= 0.0 or not np.isfinite(total):
            raise SEMTransportBackendError("SEM physical interaction kernel has no finite energy tally.")
        kernel = kernel / total
        kernel = _radial_average_kernel(kernel)
    if kernel_stack is not None:
        stack_total = float(np.sum(kernel_stack))
        if stack_total <= 0.0 or not np.isfinite(stack_total):
            raise SEMTransportBackendError("SEM physical volume kernel has no finite energy tally.")
        kernel_stack = kernel_stack / stack_total
        kernel_stack = _radial_average_kernel_stack(kernel_stack)

    observables = SEMTransportObservables(
        backscatter_coefficient=float(np.mean(backscattered)),
        max_penetration_depth_nm=float(np.max(max_depth)),
        p99_penetration_depth_nm=float(np.quantile(max_depth, 0.99)),
        active_histories_at_limit=int(np.count_nonzero(active)),
        histories=history_count,
        steps_executed=steps_executed,
    )
    return observables, kernel, kernel_stack, raw_surface_energy_tally, raw_volume_energy_tally


class PhysicalMonteCarloSEMTransportBackend:
    """SEM transport backend using elastic scattering and Joy-Luo stopping."""

    backend_mode = "monte_carlo_physical"

    def __init__(
        self,
        params: dict,
        *,
        canvas_pitch_nm: float,
        probe_sigma_px: float,
    ) -> None:
        self.backend_mode = self.__class__.backend_mode
        self.canvas_pitch_nm = _finite_nonnegative("canvas_pitch_nm", canvas_pitch_nm, minimum=1e-12)
        self.probe_sigma_px = _finite_nonnegative("sem_probe_sigma_px", probe_sigma_px, minimum=0.0)
        self._probe_sigma_nm = self.probe_sigma_px * self.canvas_pitch_nm
        self._acceleration_kV = _finite_nonnegative(
            "sem_acceleration_kV",
            param_value(params, "sem_acceleration_kV"),
            minimum=1e-9,
        )
        self._baseline = _finite_nonnegative("sem_baseline_yield", param_value(params, "sem_baseline_yield"), minimum=0.0)
        self._edge_gain = _finite_nonnegative("sem_edge_contrast_gain", param_value(params, "sem_edge_contrast_gain"), minimum=0.0)
        self._bulk_gain = _finite_nonnegative("sem_bulk_contrast_gain", param_value(params, "sem_bulk_contrast_gain"), minimum=0.0)
        self._topography_gain = _finite_nonnegative(
            "sem_topography_contrast_gain",
            param_value(params, "sem_topography_contrast_gain"),
            minimum=0.0,
        )
        self._detector_acceptance = _finite_nonnegative(
            "sem_detector_acceptance",
            param_value(params, "sem_detector_acceptance"),
            minimum=0.0,
        )
        self._takeoff_angle_deg = _finite_nonnegative(
            "sem_detector_takeoff_angle_deg",
            param_value(params, "sem_detector_takeoff_angle_deg"),
            minimum=0.0,
        )
        self._escape_depth_nm = _finite_nonnegative(
            "sem_escape_depth_nm",
            param_value(params, "sem_escape_depth_nm"),
            minimum=1e-12,
        )
        self._source_exponent = _finite_nonnegative(
            "sem_transport_source_exponent",
            param_value(params, "sem_transport_source_exponent"),
            minimum=0.05,
        )
        self._topography_source_exponent = _finite_nonnegative(
            "sem_transport_topography_exponent",
            param_value(params, "sem_transport_topography_exponent"),
            minimum=0.05,
        )
        self._beam_current_nA = _finite_nonnegative("sem_beam_current_nA", param_value(params, "sem_beam_current_nA"), minimum=0.0)
        self._dwell_time_us = _finite_nonnegative("sem_dwell_time_us", param_value(params, "sem_dwell_time_us"), minimum=0.0)
        self._electrons_per_pixel_reference = SemSettings.from_params(params).electrons_per_pixel
        self._trajectory_count = int(param_value(params, "sem_monte_carlo_trajectories"))
        if self._trajectory_count <= 0:
            raise SEMTransportBackendError("PARAMS['sem_monte_carlo_trajectories'] must be positive.")
        self._seed = resolved_sem_monte_carlo_seed(params)
        self._max_steps = int(param_value(params, "sem_physical_max_steps"))
        if self._max_steps <= 0:
            raise SEMTransportBackendError("PARAMS['sem_physical_max_steps'] must be positive.")
        self._energy_cutoff_keV = _finite_nonnegative(
            "sem_physical_energy_cutoff_keV",
            param_value(params, "sem_physical_energy_cutoff_keV"),
            minimum=1e-12,
        )
        self._elastic_model = normalize_sem_physical_elastic_model(
            param_value(params, "sem_physical_elastic_model")
        )
        self._source_representation = str(param_value(params, "sem_source_representation")).strip().lower()
        if self._source_representation not in {"projected", "volume"}:
            raise SEMTransportBackendError("PARAMS['sem_source_representation'] must be 'projected' or 'volume'.")
        kernel_raw = param_value(params, "sem_monte_carlo_kernel_size_px")
        self._kernel_size_px: int | None
        if kernel_raw is None:
            self._kernel_size_px = None
        else:
            kernel_size = int(kernel_raw)
            if kernel_size <= 0:
                raise SEMTransportBackendError("PARAMS['sem_monte_carlo_kernel_size_px'] must be positive.")
            if kernel_size % 2 == 0:
                kernel_size += 1
            self._kernel_size_px = kernel_size
        self._volume_slices = int(param_value(params, "sem_volume_slices"))
        if self._volume_slices <= 0:
            raise SEMTransportBackendError("PARAMS['sem_volume_slices'] must be positive.")
        slice_raw = param_value(params, "sem_volume_slice_thickness_nm")
        if slice_raw is None:
            self._volume_slice_thickness_nm = max(
                float(param_value(params, "sem_interaction_volume_nm")) / float(self._volume_slices),
                1e-9,
            )
        else:
            self._volume_slice_thickness_nm = float(slice_raw)
            if not np.isfinite(self._volume_slice_thickness_nm) or self._volume_slice_thickness_nm <= 0.0:
                raise SEMTransportBackendError("PARAMS['sem_volume_slice_thickness_nm'] must be positive when supplied.")
        direction_raw = param_value(params, "sem_detector_direction_xy")
        direction = np.asarray(direction_raw, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
        self._detector_direction_xy = direction / norm
        self._kernel_cache_by_material: dict[SEMMaterialChannelKey, SEMPhysicalKernelBundle] = {}

    def _kernel_size_for_material(self, material: SEMTransportMaterial) -> int:
        if self._kernel_size_px is not None:
            return self._kernel_size_px
        range_nm = kanaya_okayama_range_nm(self._acceleration_kV, material)
        radius_px = int(np.ceil((3.0 * self.probe_sigma_px) + (range_nm / self.canvas_pitch_nm) + 4.0))
        return max(9, 2 * radius_px + 1)

    def _bundle_for_material(
        self,
        key: SEMMaterialChannelKey,
        *,
        require_volume: bool,
    ) -> SEMPhysicalKernelBundle:
        cached = self._kernel_cache_by_material.get(key)
        if cached is not None and (not require_volume or cached.kernel_stack is not None):
            return cached
        material = sem_transport_material_from_channel_key(key)
        kernel_size = self._kernel_size_for_material(material)
        observables, kernel, kernel_stack, surface_tally, volume_tally = simulate_sem_transport_observables(
            material=material,
            acceleration_keV=self._acceleration_kV,
            histories=self._trajectory_count,
            seed=self._seed,
            max_steps=self._max_steps,
            cutoff_keV=self._energy_cutoff_keV,
            probe_sigma_nm=self._probe_sigma_nm,
            canvas_pitch_nm=self.canvas_pitch_nm,
            kernel_size_px=kernel_size,
            escape_depth_nm=self._escape_depth_nm,
            volume_slices=(
                self._volume_slices if require_volume else None
            ),
            volume_slice_thickness_nm=(
                self._volume_slice_thickness_nm
                if require_volume
                else None
            ),
            elastic_model=self._elastic_model,
        )
        if kernel is None:
            raise SEMTransportBackendError("SEM physical interaction kernel was not generated.")
        normalization_energy = max(float(observables.histories) * self._acceleration_kV, 1e-30)
        surface_energy_fraction = max(float(surface_tally) / normalization_energy, 0.0)
        volume_energy_fraction = max(float(volume_tally) / normalization_energy, 0.0)
        bundle = SEMPhysicalKernelBundle(
            material_key=key,
            observables=observables,
            kernel=kernel,
            kernel_stack=kernel_stack,
            raw_surface_energy_tally=surface_tally,
            raw_volume_energy_tally=volume_tally,
            surface_escape_energy_fraction_per_primary=surface_energy_fraction,
            volume_escape_energy_fraction_per_primary=volume_energy_fraction,
            kernel_size_px=kernel_size,
        )
        self._kernel_cache_by_material[key] = bundle
        return bundle

    def _interaction_kernel_stack(self, key: SEMMaterialChannelKey, num_slices: int) -> np.ndarray:
        bundle = self._bundle_for_material(key, require_volume=True)
        if bundle.kernel_stack is None:
            raise SEMTransportBackendError("SEM physical volume interaction kernel was not generated.")
        if int(num_slices) == bundle.kernel_stack.shape[0]:
            return bundle.kernel_stack
        if int(num_slices) <= 0:
            raise SEMTransportBackendError("SEM physical source stack must contain at least one slice.")
        source_z = np.linspace(0.0, 1.0, bundle.kernel_stack.shape[0])
        target_z = np.linspace(0.0, 1.0, int(num_slices))
        out = np.empty((int(num_slices), *bundle.kernel_stack.shape[-2:]), dtype=float)
        flat = bundle.kernel_stack.reshape(bundle.kernel_stack.shape[0], -1)
        for idx in range(flat.shape[1]):
            out.reshape(int(num_slices), -1)[:, idx] = np.interp(target_z, source_z, flat[:, idx])
        total = float(np.sum(out))
        if total <= 0.0 or not np.isfinite(total):
            raise SEMTransportBackendError("Resampled SEM physical volume kernel has no finite energy.")
        return out / total

    def _electrons_from_beam_current(self) -> float | None:
        return _electrons_from_beam_current(self._beam_current_nA, self._dwell_time_us)

    def electrons_per_pixel(self) -> float:
        return self._electrons_from_beam_current() or self._electrons_per_pixel_reference

    def _detector_geometry_gain(self) -> float:
        return _detector_takeoff_acceptance_gain(
            self._detector_acceptance,
            self._takeoff_angle_deg,
        )

    def _material_response(self, source: np.ndarray, key: SEMMaterialChannelKey) -> np.ndarray:
        source_positive = np.maximum(np.asarray(source, dtype=float), 0.0)
        if self._source_exponent != 1.0:
            source_positive = np.power(source_positive, self._source_exponent)
        return key.se_yield_coefficient * source_positive

    def _kernel_blur(self, arr: np.ndarray, key: SEMMaterialChannelKey) -> np.ndarray:
        bundle = self._bundle_for_material(key, require_volume=False)
        return np.maximum(_fft_convolve_centered(arr, bundle.kernel), 0.0)

    def _kernel_blur_volume(self, source_stack: np.ndarray, key: SEMMaterialChannelKey) -> np.ndarray:
        stack = np.asarray(source_stack, dtype=float)
        if stack.ndim != 3:
            raise SEMTransportBackendError(f"SEM volume source must have shape (z, y, x); got ndim={stack.ndim}.")
        kernels = self._interaction_kernel_stack(key, stack.shape[0])
        out = np.zeros(stack.shape[-2:], dtype=float)
        for idx in range(stack.shape[0]):
            out += _fft_convolve_centered(stack[idx], kernels[idx])
        return np.maximum(out, 0.0)

    def _edge_volume(self, source_stack: np.ndarray, key: SEMMaterialChannelKey) -> np.ndarray:
        stack = np.asarray(source_stack, dtype=float)
        kernels = self._interaction_kernel_stack(key, stack.shape[0])
        out = np.zeros(stack.shape[-2:], dtype=float)
        for idx in range(stack.shape[0]):
            out += _fft_convolve_centered(_gradient_magnitude(stack[idx]), kernels[idx])
        return np.maximum(out, 0.0)

    def _topography_term(self, source: np.ndarray, key: SEMMaterialChannelKey) -> np.ndarray:
        if self._topography_gain <= 0.0:
            return np.zeros_like(source)
        gx, gy = _gradient_components(source)
        directed = gy * self._detector_direction_xy[1] + gx * self._detector_direction_xy[0]
        topo = np.abs(directed)
        if self._topography_source_exponent != 1.0:
            topo = np.power(topo, self._topography_source_exponent)
        return self._topography_gain * self._kernel_blur(topo, key)

    def _transport_one_channel(
        self,
        key: SEMMaterialChannelKey,
        channel_source: np.ndarray,
    ) -> np.ndarray:
        src = np.asarray(channel_source, dtype=float)
        if not np.all(np.isfinite(src)):
            raise FloatingPointError(
                f"SEM physical Monte Carlo source for {key.material_name!r} contains non-finite values."
            )
        source_model = self._material_response(src, key)
        if source_model.ndim == 3:
            bulk = self._kernel_blur_volume(source_model, key)
            edge = self._edge_volume(source_model, key)
            topography_source = np.max(source_model, axis=0)
        elif source_model.ndim == 2:
            bulk = self._kernel_blur(source_model, key)
            edge = self._kernel_blur(_gradient_magnitude(source_model), key)
            topography_source = source_model
        else:
            raise SEMTransportBackendError(
                f"SEM physical Monte Carlo channel source must be 2D or 3D; got shape {source_model.shape!r}."
            )
        transport = (
            self._bulk_gain * bulk
            + self._edge_gain * edge
            + self._topography_term(topography_source, key)
        )
        bundle = self._bundle_for_material(key, require_volume=(source_model.ndim == 3))
        energy_scale = (
            bundle.volume_escape_energy_fraction_per_primary
            if source_model.ndim == 3 and bundle.volume_escape_energy_fraction_per_primary > 0.0
            else bundle.surface_escape_energy_fraction_per_primary
        )
        return energy_scale * transport

    def yield_from_source(self, source, *, baseline: float = 0.0) -> np.ndarray:
        if not isinstance(source, SEMMaterialSourceCanvas):
            raise SEMTransportBackendError(
                "Physical SEM transport requires SEMMaterialSourceCanvas so material identity is preserved."
            )
        output = np.zeros(source.shape[-2:], dtype=float)
        for key, channel_source in source.channels.items():
            output += self._transport_one_channel(key, channel_source)
        output = np.maximum(float(baseline) + self._detector_geometry_gain() * output, 0.0)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("SEM physical Monte Carlo backend produced non-finite yield map.")
        return output

    def contrast_from_source(self, source) -> np.ndarray:
        if not isinstance(source, SEMMaterialSourceCanvas):
            raise SEMTransportBackendError(
                "Physical SEM contrast requires SEMMaterialSourceCanvas so material identity is preserved."
            )
        return self.yield_from_source(source, baseline=0.0)

    def guard_radius_pixels(self) -> int:
        if self._kernel_cache_by_material:
            max_size = max(bundle.kernel_size_px for bundle in self._kernel_cache_by_material.values())
        elif self._kernel_size_px is not None:
            max_size = self._kernel_size_px
        else:
            max_size = int(
                np.ceil(
                    2.0
                    * (
                        3.0 * self.probe_sigma_px
                        + float(self._volume_slices * self._volume_slice_thickness_nm) / self.canvas_pitch_nm
                        + 4.0
                    )
                    + 1.0
                )
            )
            if max_size % 2 == 0:
                max_size += 1
        return int(np.ceil(0.5 * max_size + 2.0))

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        raw = params or {}
        material_rows = []
        for key, bundle in self._kernel_cache_by_material.items():
            material = sem_transport_material_from_channel_key(key)
            material_rows.append(
                {
                    "material_name": key.material_name,
                    "atomic_number": key.atomic_number,
                    "atomic_weight_g_mol": key.atomic_weight_g_mol,
                    "density_g_cm3": key.density_g_cm3,
                    "se_yield_coefficient": key.se_yield_coefficient,
                    "kernel_size_px": bundle.kernel_size_px,
                    "kernel_peak": float(np.max(bundle.kernel)),
                    "raw_surface_energy_tally_keV_weighted": bundle.raw_surface_energy_tally,
                    "raw_volume_energy_tally_keV_weighted": bundle.raw_volume_energy_tally,
                    "surface_escape_energy_fraction_per_primary": bundle.surface_escape_energy_fraction_per_primary,
                    "volume_escape_energy_fraction_per_primary": bundle.volume_escape_energy_fraction_per_primary,
                    "observables": bundle.observables.to_dict(),
                    "kanaya_okayama_range_nm": kanaya_okayama_range_nm(self._acceleration_kV, material),
                    "elastic_cross_section_cm2": float(
                        elastic_cross_section_cm2(
                            self._acceleration_kV,
                            material,
                            elastic_model=self._elastic_model,
                        )
                    ),
                    "reuter_backscatter_coefficient_20kev": (
                        reuter_backscatter_coefficient_20kev(key.atomic_number)
                        if abs(self._acceleration_kV - 20.0) <= 1e-9
                        else None
                    ),
                }
            )
        meta = SEMTransportMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level="physics_based",
            backend_name=self.backend_mode,
            equations_or_model_family="elastic_scattering_joy_luo_monte_carlo_sem_transport",
            implemented_approximation_level=f"{self._elastic_model}_joy_luo_material_resolved_transport",
            native_operating_assumptions=(
                "material-resolved elastic scattering, Joy-Luo continuous "
                "slowing-down energy loss, near-surface secondary-electron escape weighting, "
                "and one precomputed seeded interaction kernel per source material"
            ),
        ).to_dict()
        meta.update(
            {
                "kind": "sem_monte_carlo_physical_transport",
                "sem_backend": self.backend_mode,
                "backend_mode": self.backend_mode,
                "backend_fidelity_level": "physics_based",
                "forward_observable": "elastic-scattering/Joy-Luo SEM secondary-electron transport kernel",
                "validation_status": "diagnostic_only",
                "fidelity_label": f"sem_{self._elastic_model}_joy_luo_physics_unvalidated",
                "reference_backend_metadata": {
                    "validation_script": "validation/sem_transport_validation.py",
                    "claim_maturity_gate": "diagnostic_only_until_full_tolerance_run_passes",
                    "primary_reference": "20 keV empirical backscatter-coefficient fit",
                    "secondary_reference": "Kanaya-Okayama electron range",
                    "elastic_cross_section_reference": (
                        "Browning empirical Mott surrogate"
                        if self._elastic_model == "mott_browning"
                        else "screened Rutherford"
                    ),
                },
                "material_model_scope": "material_resolved_sem_source_channels",
                "sem_physical_elastic_model": self._elastic_model,
                "sem_physical_material_channels": material_rows,
                "sem_physical_energy_cutoff_keV": self._energy_cutoff_keV,
                "acceleration_kV": self._acceleration_kV,
                "beam_current_nA": self._beam_current_nA,
                "dwell_time_us": self._dwell_time_us,
                "electrons_per_pixel": self.electrons_per_pixel(),
                "probe_sigma_canvas_pixels": self.probe_sigma_px,
                "probe_sigma_nm": self._probe_sigma_nm,
                "monte_carlo_trajectories": self._trajectory_count,
                "sem_physical_max_steps": self._max_steps,
                "monte_carlo_seed": self._seed,
                "monte_carlo_kernel_size_px": self._kernel_size_px,
                "sem_physical_observables": material_rows,
                "source_representation": self._source_representation,
                "sem_source_representation": self._source_representation,
                "sem_volume_slices": self._volume_slices,
                "sem_volume_slice_thickness_nm": self._volume_slice_thickness_nm,
                "sem_volume_depth_nm": self._volume_slices * self._volume_slice_thickness_nm,
                "escape_depth_nm": self._escape_depth_nm,
                "detector_takeoff_angle_deg": self._takeoff_angle_deg,
                "detector_acceptance": self._detector_acceptance,
                "detector_direction_xy": [float(v) for v in self._detector_direction_xy],
                "comparison_contract_id": str(raw.get("comparison_contract_id", "Contract-NR")),
                "artifact_provenance_id": raw.get("artifact_provenance_id", None),
            }
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=raw,
            backend_name=self.backend_mode,
            equations_or_model_family="elastic_scattering_joy_luo_monte_carlo_sem_transport",
            implemented_approximation_level=f"{self._elastic_model}_joy_luo_material_resolved_transport",
            native_operating_assumptions=meta["native_operating_assumptions"],
            comparison_contract_id=str(raw.get("comparison_contract_id", "Contract-NR")),
            artifact_provenance_id=raw.get("artifact_provenance_id", None),
        )


__all__ = [
    "PhysicalMonteCarloSEMTransportBackend",
    "SEMPhysicalKernelBundle",
    "SEMTransportObservables",
    "browning_mott_cross_section_cm2",
    "elastic_cross_section_cm2",
    "elastic_mean_free_path_nm",
    "joy_luo_stopping_power_keV_per_nm",
    "kanaya_okayama_range_nm",
    "normalize_sem_physical_elastic_model",
    "reuter_backscatter_coefficient_20kev",
    "sample_elastic_scattering_cos_theta",
    "screened_rutherford_cross_section_cm2",
    "sem_transport_material_from_channel_key",
    "simulate_sem_transport_observables",
]
