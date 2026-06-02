"""roughness substrate-pattern helpers."""

from __future__ import annotations

from ._shared import (
    cv2,
    math,
    np,
    os,
)

def _normalize_to_unit_std(field: np.ndarray) -> np.ndarray:
    """
    Center a floating field and normalize it to unit standard deviation when
    possible. Degenerate fields return all zeros.
    """
    field = np.asarray(field, dtype=float)
    centered = field - float(np.mean(field))
    std = float(np.std(centered))
    if std <= 1e-12:
        return np.zeros_like(centered)
    return centered / std

def _resize_complex_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Resize a complex field to a target (height, width) using separable real/imaginary
    interpolation.
    """
    target_h, target_w = int(shape[0]), int(shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target shape dimensions must be positive.")
    field = np.asarray(field)
    if field.shape == (target_h, target_w):
        return field.astype(np.complex128, copy=False)
    real_part = cv2.resize(
        np.real(field).astype(float),
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )
    imag_part = cv2.resize(
        np.imag(field).astype(float),
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )
    return (real_part + 1j * imag_part).astype(np.complex128)

def _load_roughness_reference_field(
    raw_field,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Load a user-supplied roughness/speckle reference field.

    The source may be a real or complex NumPy-like array, a complex/real
    two-channel representation, or a path to a supported raster format.
    """
    raw = raw_field
    if raw is None:
        raise ValueError(
            "sample_environment_pattern_roughness_source must be provided when "
            "roughness model is 'source_matched'."
        )

    if isinstance(raw, (str, bytes, os.PathLike)):
        path = os.fspath(raw)
        ext = os.path.splitext(path)[1].strip().lower()
        if not os.path.exists(path):
            raise ValueError(f"sample_environment_pattern_roughness_source path not found: {path!r}.")
        if ext == ".npy":
            raw = np.load(path)
        else:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(
                    "sample_environment_pattern_roughness_source could not be read by cv2: "
                    f"{path!r}."
                )
            raw = image

    arr = np.asarray(raw)
    if arr.size == 0:
        raise ValueError(
            "sample_environment_pattern_roughness_source must contain a non-empty array."
        )

    if arr.ndim == 0:
        raise ValueError("sample_environment_pattern_roughness_source must be 2D (or 2-channel 3D).")

    if arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = np.asarray(arr[..., 0], dtype=float) + 1j * np.asarray(arr[..., 1], dtype=float)
        elif arr.shape[-1] in (3, 4):
            if arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                "sample_environment_pattern_roughness_source with ndim==3 must be HxWx2 "
                "for complex [real, imag] channel format."
            )

    if not np.iscomplexobj(arr):
        arr = np.asarray(arr, dtype=float)

    arr = arr.astype(np.complex128, copy=False)
    return _resize_complex_field(arr, target_shape)

def _normalize_roughness_field(field: np.ndarray) -> np.ndarray:
    """
    Normalize a roughness field so multiplicative amplitude has unit mean.
    """
    amp = np.abs(field)
    amp_mean = float(np.mean(amp))
    if amp_mean <= 0.0 or not np.isfinite(amp_mean):
        return np.ones(field.shape, dtype=np.complex128)
    amp = amp / amp_mean
    phase = np.zeros_like(amp)
    finite = amp > 0.0
    if np.any(finite):
        phase[finite] = np.angle(field[finite])
    return (amp * np.exp(1j * phase)).astype(np.complex128)

def generate_empirical_background_field(
    params: dict,
    final_fov_shape: tuple,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a dimensionless, mean-one, spatially correlated nuisance field.

    This field represents residual illumination / flat-field variation,
    detector-offset nonuniformity, and other slowly varying empirical
    background terms. It is intentionally modeled as a smooth nuisance field,
    not as a latent population of out-of-focus particles.
    """
    shape = (int(final_fov_shape[0]), int(final_fov_shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("final_fov_shape must contain positive dimensions.")
    rng = np.random.default_rng() if rng is None else rng

    enabled = bool(params.get("empirical_background_enabled", False))
    if not enabled:
        return np.ones(shape, dtype=float)

    model = str(
        params.get("empirical_background_model", "multiscale_gaussian_field")
    ).strip().lower()
    if model not in ("multiscale_gaussian_field", "none"):
        raise ValueError(
            "Unsupported empirical_background_model "
            f"'{params.get('empirical_background_model')}'."
        )
    if model == "none":
        return np.ones(shape, dtype=float)

    relative_std = float(params.get("empirical_background_relative_std", 0.03))
    gradient_strength = float(
        params.get("empirical_background_gradient_relative_strength", 0.0)
    )
    if relative_std < 0.0:
        raise ValueError("empirical_background_relative_std must be non-negative.")
    if gradient_strength < 0.0:
        raise ValueError(
            "empirical_background_gradient_relative_strength must be non-negative."
        )

    scales = params.get("empirical_background_scales_px", [16.0, 64.0, 256.0])
    weights = params.get("empirical_background_scale_weights", [0.4, 0.35, 0.25])
    if len(scales) != len(weights):
        raise ValueError(
            "empirical_background_scales_px and "
            "empirical_background_scale_weights must have the same length."
        )
    if len(scales) == 0:
        return np.ones(shape, dtype=float)

    field = np.zeros(shape, dtype=float)
    for sigma_px, weight in zip(scales, weights):
        sigma = float(sigma_px)
        layer_weight = float(weight)
        if sigma <= 0.0 or layer_weight == 0.0:
            continue
        white = rng.normal(loc=0.0, scale=1.0, size=shape)
        smooth = cv2.GaussianBlur(
            white,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        field += layer_weight * _normalize_to_unit_std(smooth)

    field = _normalize_to_unit_std(field)

    if gradient_strength > 0.0:
        yy, xx = np.indices(shape, dtype=float)
        x_norm = (xx - float(np.mean(xx))) / max(float(shape[1] - 1), 1.0)
        y_norm = (yy - float(np.mean(yy))) / max(float(shape[0] - 1), 1.0)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        gradient = math.cos(angle) * x_norm + math.sin(angle) * y_norm
        field += gradient_strength / max(relative_std, 1e-12) * _normalize_to_unit_std(
            gradient
        )
        field = _normalize_to_unit_std(field)

    if relative_std == 0.0:
        return np.ones(shape, dtype=float)

    nuisance = 1.0 + relative_std * field
    nuisance = np.clip(nuisance, 1e-6, None)
    mean_val = float(np.mean(nuisance))
    if mean_val > 0.0:
        nuisance /= mean_val
    return nuisance.astype(float)

def resize_empirical_background_field(
    nuisance_final: np.ndarray,
    target_shape: tuple,
) -> np.ndarray:
    """
    Resize a final-resolution empirical background field to another grid.
    """
    nuisance_final = np.asarray(nuisance_final, dtype=float)
    target = cv2.resize(
        nuisance_final,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(target, 1e-6, None).astype(float)

def _generate_correlated_unit_field(
    shape: tuple,
    correlation_pixels: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a zero-mean unit-variance field with optional spatial correlation.
    """
    raw = rng.normal(size=shape, loc=0.0, scale=1.0)
    correlation = float(correlation_pixels)
    if correlation > 0.0:
        raw = cv2.GaussianBlur(
            raw.astype(float),
            ksize=(0, 0),
            sigmaX=correlation,
            sigmaY=correlation,
            borderType=cv2.BORDER_REFLECT,
        )
    return _normalize_to_unit_std(np.asarray(raw, dtype=float))

def generate_sample_environment_roughness_field(
    params: dict,
    shape: tuple,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a mean-one multiplicative roughness/speckle field.

    The field is complex-valued and can include both amplitude and phase
    perturbations controlled by:
    - sample_environment_pattern_roughness_model
    - sample_environment_pattern_roughness_amplitude
    - sample_environment_pattern_roughness_correlation_pixels
    - sample_environment_pattern_roughness_phase_std
    """
    shape = (int(shape[0]), int(shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain positive dimensions for roughness field.")
    if rng is None:
        rng = np.random.default_rng()

    roughness_model_raw = params.get("sample_environment_pattern_roughness_model", "none")
    roughness_model = str(roughness_model_raw).strip().lower()
    if roughness_model not in ("none", "static", "flicker", "source_matched"):
        raise ValueError(
            "Unsupported sample_environment_pattern_roughness_model "
            f"'{params.get('sample_environment_pattern_roughness_model')}'."
        )

    if roughness_model == "none":
        return np.ones(shape, dtype=np.complex128)

    if roughness_model == "source_matched":
        source_field = _load_roughness_reference_field(
            params.get("sample_environment_pattern_roughness_source"),
            shape,
        )
        roughness_field = _normalize_roughness_field(source_field)
        roughness_amplitude = float(
            params.get("sample_environment_pattern_roughness_amplitude", 0.0)
        )
        if roughness_amplitude != 0.0:
            correlation_pixels = float(
                params.get("sample_environment_pattern_roughness_correlation_pixels", 4.0)
            )
            correlation_pixels = max(correlation_pixels, 0.0)
            amp_noise = _generate_correlated_unit_field(shape, correlation_pixels, rng)
            roughness_field = roughness_field * np.exp(roughness_amplitude * amp_noise)
            roughness_field = _normalize_roughness_field(roughness_field)
        phase_std = float(
            params.get("sample_environment_pattern_roughness_phase_std", 0.0)
        )
        if phase_std > 0.0:
            phase_correlation_pixels = float(
                params.get("sample_environment_pattern_roughness_correlation_pixels", 4.0)
            )
            phase_correlation_pixels = max(phase_correlation_pixels, 0.0)
            phase_noise = _generate_correlated_unit_field(shape, phase_correlation_pixels, rng)
            roughness_field = roughness_field * np.exp(1j * phase_std * phase_noise)
        return roughness_field.astype(np.complex128)

    roughness_amplitude = float(
        params.get("sample_environment_pattern_roughness_amplitude", 0.0)
    )
    if roughness_amplitude <= 0.0:
        return np.ones(shape, dtype=np.complex128)

    correlation_pixels = float(
        params.get("sample_environment_pattern_roughness_correlation_pixels", 4.0)
    )
    correlation_pixels = max(correlation_pixels, 0.0)

    phase_std = float(params.get("sample_environment_pattern_roughness_phase_std", 0.0))

    amp_noise = _generate_correlated_unit_field(shape, correlation_pixels, rng)
    amplitude = np.exp(roughness_amplitude * amp_noise)
    amplitude_mean = float(np.mean(amplitude))
    if amplitude_mean <= 0.0:
        amplitude = np.ones(shape, dtype=float)
    else:
        amplitude = amplitude / amplitude_mean

    phase = np.ones(shape, dtype=float)
    if phase_std > 0.0:
        phase_noise = _generate_correlated_unit_field(shape, correlation_pixels, rng)
        phase = np.exp(1j * phase_std * phase_noise).astype(np.complex128)

    return (amplitude * phase).astype(np.complex128)

__all__ = ['generate_empirical_background_field', 'resize_empirical_background_field', 'generate_sample_environment_roughness_field']
