"""Analysis-noise likelihood contracts passed across Fisher seams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AnalysisNoiseModel:
    diagonal_variance: Any
    measurement_domain: str
    signal_units: str
    noise_variance_units: str
    covariance_kind: str = "independent_pixels"
    row_correlated_variance: float = 0.0
    row_correlated_component_variances: tuple[float, ...] = ()
    row_correlated_couplings: Any = None
    fourier_sideband_raw_variance: Any = None
    fourier_sideband_mask: Any = None
    fourier_sideband_shift: tuple[int, int] = (0, 0)
    fourier_sideband_phase_correction: Any = None
    fourier_sideband_output_normalization: Any = None
    fourier_sideband_output_conjugate: bool = False
    fourier_sideband_operator: str = ""
    safe_for_ordering: bool = True
    safe_for_fusion: bool = True
    status_reason: str = ""

    def variance_array(self) -> np.ndarray:
        return np.asarray(self.diagonal_variance, dtype=float)

    def require_safe_for_fisher(self, *, context: str) -> None:
        if not self.safe_for_ordering:
            raise ValueError(
                f"{context} received an AnalysisNoiseModel that is not safe for "
                f"Fisher ranking: {self.status_reason or self.covariance_kind}."
            )


INDEPENDENT_PIXEL_NOISE_CONTRACT_ID = "syniscopy-independent-pixel-fisher-noise-v1"


@dataclass(frozen=True)
class IndependentPixelNoiseModel:
    """Explicit diagonal Fisher likelihood for independent-pixel noise.

    This is not a report summary. It is a physical assertion that the supplied
    diagonal variance is the complete covariance operator in the same analysis
    basis as the image derivatives. Raw ndarrays/scalars are intentionally not a
    Fisher likelihood, because a diagonal projection of a structured likelihood
    has the same shape but lacks the off-diagonal covariance needed for CRLB.
    """

    diagonal_variance: Any
    measurement_domain: str = "contrast"
    signal_units: str = "contrast"
    noise_variance_units: str = "contrast_squared"
    covariance_kind: str = "independent_pixels"
    safe_for_ordering: bool = True
    safe_for_fusion: bool = True
    status_reason: str = "explicit independent-pixel Fisher likelihood"
    contract_id: str = INDEPENDENT_PIXEL_NOISE_CONTRACT_ID

    def variance_array(self) -> np.ndarray:
        return np.asarray(self.diagonal_variance, dtype=float)

    def require_safe_for_fisher(self, *, context: str) -> None:
        if str(self.covariance_kind) != "independent_pixels":
            raise ValueError(
                f"{context} received IndependentPixelNoiseModel with "
                f"covariance_kind={self.covariance_kind!r}; expected 'independent_pixels'."
            )
        if not self.safe_for_ordering:
            raise ValueError(
                f"{context} received an IndependentPixelNoiseModel that is not safe "
                f"for Fisher ranking: {self.status_reason}."
            )


def _default_noise_variance_units(signal_units: str) -> str:
    unit = str(signal_units or "contrast").strip() or "contrast"
    return f"{unit}_squared"


def independent_pixel_noise_model(
    variance: Any,
    *,
    measurement_domain: str = "contrast",
    signal_units: str = "contrast",
    noise_variance_units: str | None = None,
    safe_for_ordering: bool = True,
    safe_for_fusion: bool = True,
    status_reason: str = "explicit independent-pixel Fisher likelihood",
    context: str = "independent-pixel Fisher likelihood",
) -> IndependentPixelNoiseModel:
    """Construct an explicit independent-pixel Fisher likelihood.

    The constructor is the only sanctioned way to turn a scalar/array diagonal
    variance into a Fisher input. It makes the independence assumption visible at
    the code seam so a report-only diagonal summary cannot masquerade as a full
    covariance model.
    """
    arr = np.asarray(variance, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{context} variance is empty.")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{context} variance must contain only positive finite values.")
    diagonal_variance: Any = float(arr) if arr.shape == () else arr
    return IndependentPixelNoiseModel(
        diagonal_variance=diagonal_variance,
        measurement_domain=str(measurement_domain),
        signal_units=str(signal_units),
        noise_variance_units=(
            _default_noise_variance_units(signal_units)
            if noise_variance_units is None or str(noise_variance_units).strip() == ""
            else str(noise_variance_units)
        ),
        covariance_kind="independent_pixels",
        safe_for_ordering=bool(safe_for_ordering),
        safe_for_fusion=bool(safe_for_fusion),
        status_reason=str(status_reason),
    )


def fourier_sideband_demodulated_noise_model(
    *,
    diagonal_variance: Any,
    raw_variance: Any,
    sideband_mask: Any,
    sideband_shift: tuple[int, int],
    sideband_phase_correction: Any = None,
    sideband_output_normalization: Any = None,
    sideband_output_conjugate: bool = False,
    raw_row_correlated_variance: float = 0.0,
    measurement_domain: str = "demodulated_complex_field",
    signal_units: str = "complex_detector_count",
    noise_variance_units: str = "complex_detector_count_squared",
    safe_for_ordering: bool = True,
    safe_for_fusion: bool = True,
    status_reason: str = "Fourier-sideband demodulated covariance propagated from raw detector counts",
    context: str = "Fourier-sideband demodulated Fisher likelihood",
) -> AnalysisNoiseModel:
    """Construct the structured likelihood for off-axis DHM demodulation.

    The raw detector frame has independent Poisson/readout noise in count
    space.  Fourier sideband extraction is a linear projection that makes the
    reconstructed complex field spatially correlated, so the diagonal variance
    here is only a report summary.  Fisher precision must use the sideband
    operator payload.
    """

    diagonal = np.asarray(diagonal_variance, dtype=float)
    raw = np.asarray(raw_variance, dtype=float)
    mask = np.asarray(sideband_mask, dtype=bool)
    if diagonal.ndim != 2:
        raise ValueError(f"{context} diagonal_variance must be a 2D image; got {diagonal.shape}.")
    if raw.shape != diagonal.shape:
        raise ValueError(
            f"{context} raw_variance shape {raw.shape} must match diagonal_variance "
            f"shape {diagonal.shape}."
        )
    if mask.shape != diagonal.shape:
        raise ValueError(
            f"{context} sideband_mask shape {mask.shape} must match diagonal_variance "
            f"shape {diagonal.shape}."
        )
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError(f"{context} diagonal_variance must contain only positive finite values.")
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(f"{context} raw_variance must contain only positive finite values.")
    if not np.any(mask):
        raise ValueError(f"{context} sideband_mask must select at least one Fourier coefficient.")
    shift = tuple(int(v) for v in sideband_shift)
    if len(shift) != 2:
        raise ValueError(f"{context} sideband_shift must be a (dy, dx) tuple.")
    phase_correction = None
    if sideband_phase_correction is not None:
        phase_correction = np.asarray(sideband_phase_correction, dtype=np.complex128)
        if phase_correction.shape != diagonal.shape:
            raise ValueError(
                f"{context} sideband_phase_correction shape {phase_correction.shape} "
                f"must match diagonal_variance shape {diagonal.shape}."
            )
        if np.any(~np.isfinite(phase_correction.real)) or np.any(~np.isfinite(phase_correction.imag)):
            raise ValueError(f"{context} sideband_phase_correction must contain only finite values.")
        if not np.allclose(np.abs(phase_correction), 1.0, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"{context} sideband_phase_correction must be unit magnitude.")
    output_normalization = None
    if sideband_output_normalization is not None:
        output_normalization = np.asarray(sideband_output_normalization, dtype=np.complex128)
        if output_normalization.shape != diagonal.shape:
            raise ValueError(
                f"{context} sideband_output_normalization shape "
                f"{output_normalization.shape} must match diagonal_variance shape "
                f"{diagonal.shape}."
            )
        if (
            np.any(~np.isfinite(output_normalization.real))
            or np.any(~np.isfinite(output_normalization.imag))
        ):
            raise ValueError(f"{context} sideband_output_normalization must contain only finite values.")
        if np.any(np.abs(output_normalization) <= 0.0):
            raise ValueError(f"{context} sideband_output_normalization must be nonzero everywhere.")
    row_variance = float(raw_row_correlated_variance)
    if not np.isfinite(row_variance) or row_variance < 0.0:
        raise ValueError(f"{context} raw_row_correlated_variance must be finite and non-negative.")
    return AnalysisNoiseModel(
        diagonal_variance=diagonal,
        measurement_domain=str(measurement_domain),
        signal_units=str(signal_units),
        noise_variance_units=str(noise_variance_units),
        covariance_kind="fourier_sideband_demodulated_complex_field",
        row_correlated_variance=row_variance,
        fourier_sideband_raw_variance=raw,
        fourier_sideband_mask=mask,
        fourier_sideband_shift=(int(shift[0]), int(shift[1])),
        fourier_sideband_phase_correction=phase_correction,
        fourier_sideband_output_normalization=output_normalization,
        fourier_sideband_output_conjugate=bool(sideband_output_conjugate),
        fourier_sideband_operator=(
            "raw_count_diagonal_covariance_pushed_through_fft_sideband_projection"
            "_fractional_carrier_phase_correction_and_declared_output_normalization"
        ),
        safe_for_ordering=bool(safe_for_ordering),
        safe_for_fusion=bool(safe_for_fusion),
        status_reason=str(status_reason),
    )


def fisher_noise_input_to_analysis_model(
    payload: AnalysisNoiseModel | IndependentPixelNoiseModel | Mapping[str, Any],
    *,
    context: str = "Fisher noise input",
) -> AnalysisNoiseModel | IndependentPixelNoiseModel:
    """Resolve a typed Fisher likelihood without accepting raw diagonal arrays."""
    if isinstance(payload, (AnalysisNoiseModel, IndependentPixelNoiseModel)):
        return payload
    if isinstance(payload, Mapping) and payload.get("contract_id") == INDEPENDENT_PIXEL_NOISE_CONTRACT_ID:
        return independent_pixel_noise_model(
            payload.get("diagonal_variance"),
            measurement_domain=str(payload.get("measurement_domain", "contrast")),
            signal_units=str(payload.get("signal_units", "contrast")),
            noise_variance_units=payload.get("noise_variance_units"),
            safe_for_ordering=bool(payload.get("safe_for_ordering", True)),
            safe_for_fusion=bool(payload.get("safe_for_fusion", True)),
            status_reason=str(payload.get("status_reason", "serialized independent-pixel Fisher likelihood")),
            context=context,
        )
    if isinstance(payload, Mapping):
        return analysis_noise_model_from_likelihood(payload, context=context)
    raise TypeError(
        f"{context} must be an AnalysisNoiseModel, IndependentPixelNoiseModel, "
        "or serialized likelihood mapping. Raw scalar/array variances are not "
        "a complete Fisher covariance contract; wrap independent-pixel variances "
        "with independent_pixel_noise_model(...)."
    )


@dataclass(frozen=True)
class AnalysisNoiseSummary:
    diagonal_variance: np.ndarray
    mean_diagonal_variance: float
    measurement_domain: str
    signal_units: str
    noise_variance_units: str
    covariance_kind: str
    safe_for_ordering: bool
    safe_for_fusion: bool
    status_reason: str


FISHER_LIKELIHOOD_ELIGIBILITY_CONTRACT_ID = "syniscopy-fisher-likelihood-eligibility-v1"


@dataclass(frozen=True)
class FisherLikelihoodEligibility:
    """Resolved production eligibility for Fisher/CRLB ranking and fusion.

    The analysis-noise object owns contrast-domain covariance validity, while
    detector metadata owns post-Poisson transfer validity.  Keeping this
    composition in one contract prevents report, simulation, and dynamic paths
    from independently treating a finite CRLB as production-rankable when the
    detector likelihood is still diagnostic-only.
    """

    safe_for_ordering: bool
    safe_for_fusion: bool
    detector_safe_for_report_fisher: bool
    used_covariance_fisher: bool
    safe_for_linear_fisher_variance: bool
    safe_for_covariance_fisher_variance: bool
    analysis_noise_safe_for_ordering: bool
    analysis_noise_safe_for_fusion: bool
    fisher_singular: bool
    status_reason: str
    detector_likelihood_status: str
    contract_id: str = FISHER_LIKELIHOOD_ELIGIBILITY_CONTRACT_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_for_ordering": bool(self.safe_for_ordering),
            "safe_for_fusion": bool(self.safe_for_fusion),
            "detector_safe_for_report_fisher": bool(self.detector_safe_for_report_fisher),
            "fisher_likelihood_uses_covariance": bool(self.used_covariance_fisher),
            "safe_for_linear_fisher_variance": bool(self.safe_for_linear_fisher_variance),
            "safe_for_covariance_fisher_variance": bool(self.safe_for_covariance_fisher_variance),
            "analysis_noise_safe_for_ordering": bool(self.analysis_noise_safe_for_ordering),
            "analysis_noise_safe_for_fusion": bool(self.analysis_noise_safe_for_fusion),
            "fisher_singular": bool(self.fisher_singular),
            "status_reason": str(self.status_reason),
            "detector_likelihood_status": str(self.detector_likelihood_status),
            "fisher_likelihood_eligibility_contract_id": self.contract_id,
        }


def _contract_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _contract_bool(source: Any, key: str, default: bool = False) -> bool:
    value = _contract_value(source, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _fisher_result_uses_covariance(crlb_metadata: Mapping[str, Any] | None) -> bool:
    crlb = dict(crlb_metadata or {})
    model = str(crlb.get("fisher_noise_covariance_model", "")).strip()
    if model:
        return True
    derivative_metadata = crlb.get("derivative_metadata")
    if isinstance(derivative_metadata, Mapping):
        return bool(str(derivative_metadata.get("fisher_noise_covariance_model", "")).strip())
    return False


def resolve_fisher_likelihood_eligibility(
    analysis_noise: AnalysisNoiseModel | AnalysisNoiseSummary | Mapping[str, Any],
    detector_metadata: Mapping[str, Any],
    crlb_metadata: Mapping[str, Any] | None = None,
    *,
    context: str = "Fisher/CRLB result",
) -> FisherLikelihoodEligibility:
    """Compose analysis-noise and detector-transfer safety for CRLB consumers.

    A finite Fisher matrix is not by itself a production scientific claim.  It is
    safe for report ordering/fusion only when (1) the contrast-domain likelihood
    is valid, (2) the Fisher result is nonsingular, and (3) the detector transfer
    stage is compatible with the Fisher variance model actually used.  Row
    covariance may relax the linear-diagonal detector flag, but nonlinear/static
    detector transfer remains diagnostic-only until a transfer-aware likelihood
    is implemented.
    """

    crlb = dict(crlb_metadata or {})
    detector = dict(detector_metadata or {})
    fisher_singular = bool(crlb.get("singular", crlb.get("fisher_singular", False)))
    used_covariance_fisher = _fisher_result_uses_covariance(crlb)

    analysis_ordering = _contract_bool(analysis_noise, "safe_for_ordering", True)
    analysis_fusion = _contract_bool(analysis_noise, "safe_for_fusion", analysis_ordering)
    analysis_reason = str(_contract_value(analysis_noise, "status_reason", "") or "")

    linear_safe = bool(detector.get("safe_for_linear_fisher_variance", False))
    covariance_safe = bool(
        detector.get("safe_for_covariance_fisher_variance", linear_safe)
    )
    detector_safe = bool(linear_safe or (used_covariance_fisher and covariance_safe))
    detector_status = str(detector.get("detector_likelihood_status", "") or "")

    reasons: list[str] = []
    if fisher_singular:
        reasons.append(f"{context} Fisher matrix is singular")
    if not analysis_ordering:
        reasons.append(
            "analysis contrast-noise likelihood is not safe for ordering"
            + (f": {analysis_reason}" if analysis_reason else "")
        )
    if not detector_safe:
        if used_covariance_fisher and not covariance_safe:
            reasons.append(
                "detector transfer is not safe for covariance Fisher variance"
                + (f": {detector_status}" if detector_status else "")
            )
        elif not linear_safe:
            reasons.append(
                "detector transfer is not safe for linear Fisher variance"
                + (f": {detector_status}" if detector_status else "")
            )

    safe_for_ordering = bool(analysis_ordering and detector_safe and not fisher_singular)
    safe_for_fusion = bool(analysis_fusion and detector_safe and not fisher_singular)
    status_reason = "; ".join(reasons)
    if safe_for_ordering and not safe_for_fusion:
        status_reason = (
            "analysis contrast-noise likelihood is not safe for fusion"
            + (f": {analysis_reason}" if analysis_reason else "")
        )

    return FisherLikelihoodEligibility(
        safe_for_ordering=safe_for_ordering,
        safe_for_fusion=safe_for_fusion,
        detector_safe_for_report_fisher=detector_safe,
        used_covariance_fisher=used_covariance_fisher,
        safe_for_linear_fisher_variance=linear_safe,
        safe_for_covariance_fisher_variance=covariance_safe,
        analysis_noise_safe_for_ordering=analysis_ordering,
        analysis_noise_safe_for_fusion=analysis_fusion,
        fisher_singular=fisher_singular,
        status_reason=status_reason,
        detector_likelihood_status=detector_status,
    )


def scale_analysis_noise_model(
    noise_model: AnalysisNoiseModel,
    *,
    variance_scale: float,
    measurement_domain: str | None = None,
    signal_units: str | None = None,
    noise_variance_units: str | None = None,
    context: str = "analysis noise basis transform",
) -> AnalysisNoiseModel:
    """Return ``noise_model`` transformed by a scalar signal-basis change.

    If an analysis image is rescaled by ``a`` then diagonal variances and any
    directly stored row variance scale by ``a**2``.  Row-coupling vectors scale
    by ``a`` so the covariance term ``v * u u.T`` remains in the same basis as
    the image passed to Fisher/CRLB.
    """
    scale = float(variance_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{context}: variance_scale must be positive; got {variance_scale!r}.")

    coupling_scale = float(np.sqrt(scale))
    row_couplings = noise_model.row_correlated_couplings
    if row_couplings is not None:
        row_couplings = np.asarray(row_couplings, dtype=float) * coupling_scale
    fourier_raw_variance = noise_model.fourier_sideband_raw_variance
    if fourier_raw_variance is not None:
        fourier_raw_variance = np.asarray(fourier_raw_variance, dtype=float) * scale

    return replace(
        noise_model,
        diagonal_variance=np.asarray(noise_model.diagonal_variance, dtype=float) * scale,
        row_correlated_variance=float(noise_model.row_correlated_variance) * scale,
        row_correlated_couplings=row_couplings,
        fourier_sideband_raw_variance=fourier_raw_variance,
        measurement_domain=(
            noise_model.measurement_domain if measurement_domain is None else str(measurement_domain)
        ),
        signal_units=noise_model.signal_units if signal_units is None else str(signal_units),
        noise_variance_units=(
            noise_model.noise_variance_units
            if noise_variance_units is None
            else str(noise_variance_units)
        ),
    )


def summarize_analysis_noise_model(
    noise_model: AnalysisNoiseModel,
    *,
    expected_shape: tuple[int, ...] | None = None,
    context: str = "analysis noise",
) -> AnalysisNoiseSummary:
    """Return the reportable diagonal-variance projection of a noise model.

    Fisher/CRLB callers should still receive the full ``AnalysisNoiseModel``.
    This helper is only for report/metadata fields that need a scalar or array
    diagonal-variance summary while preserving the model's units and covariance
    provenance.
    """
    if not isinstance(noise_model, AnalysisNoiseModel):
        raise TypeError(
            f"{context} expected an AnalysisNoiseModel; got {type(noise_model).__name__}."
        )
    variance = noise_model.variance_array()
    if variance.size == 0:
        raise RuntimeError(f"Noise-variance array is empty for {context}.")
    if np.any(~np.isfinite(variance)):
        raise RuntimeError(f"Noise-variance array contains non-finite values for {context}.")
    if np.any(variance <= 0.0):
        raise RuntimeError(f"Noise-variance array contains non-positive values for {context}.")
    if expected_shape is not None:
        expected = tuple(int(v) for v in expected_shape)
        if variance.shape != expected and variance.size != 1:
            raise RuntimeError(
                "Noise-variance shape does not match "
                f"{context}: {variance.shape} vs {expected}."
            )
    return AnalysisNoiseSummary(
        diagonal_variance=variance,
        mean_diagonal_variance=float(np.nanmean(variance)),
        measurement_domain=str(noise_model.measurement_domain),
        signal_units=str(noise_model.signal_units),
        noise_variance_units=str(noise_model.noise_variance_units),
        covariance_kind=str(noise_model.covariance_kind),
        safe_for_ordering=bool(noise_model.safe_for_ordering),
        safe_for_fusion=bool(noise_model.safe_for_fusion),
        status_reason=str(noise_model.status_reason),
    )


ANALYSIS_NOISE_LIKELIHOOD_SCHEMA_VERSION = "syniscopy-analysis-noise-likelihood-v1"



PHASE_LIKELIHOOD_SCHEMA_VERSION = "syniscopy-phase-likelihood-v1"


@dataclass(frozen=True)
class PhaseLikelihoodBasis:
    """Explicit photon/quanta support for phase-domain Fisher likelihoods.

    A phase image in radians is not itself a photon-count image, and QPI display
    counts are only a visualization basis.  Phase shot-noise therefore needs a
    separate detected-quanta support object so Fisher callers cannot accidentally
    infer photon statistics from phase or display-count arrays.
    """

    detected_quanta_per_pixel: Any
    visibility: Any
    readout_variance_rad2: Any = 0.0
    provenance: str = "uniform_config_scalar"
    contract_id: str = PHASE_LIKELIHOOD_SCHEMA_VERSION
    measurement_domain: str = "phase"
    signal_units: str = "radian"
    noise_variance_units: str = "radian_squared"


def _coerce_scalar_or_map(value: Any, shape: tuple[int, ...], *, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        scalar = float(arr)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{field_name} must be positive and finite; got {value!r}.")
        return np.full(shape, scalar, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{field_name} shape {arr.shape} does not match phase frame shape {shape}.")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{field_name} map must contain only positive finite values.")
    return arr


def phase_variance_from_likelihood_basis(
    basis: PhaseLikelihoodBasis,
    shape: tuple[int, ...],
    *,
    shot_noise_enabled: bool = True,
    gaussian_noise_enabled: bool = True,
    variance_floor: float = 1e-30,
) -> np.ndarray:
    """Return phase variance from an explicit detected-quanta likelihood basis.

    This is the shared phase-domain contract: shot noise scales as
    ``1 / (visibility**2 * detected_quanta_per_pixel)`` at each pixel, while
    readout/demodulation variance is already in radian-squared units.  The
    helper accepts scalar values only when the caller explicitly constructs a
    scalar likelihood basis; patterned/reference-varying renderers should pass
    maps with the same shape as the phase image.
    """

    expected_shape = tuple(int(v) for v in shape)
    if len(expected_shape) < 2:
        raise ValueError(f"phase likelihood shape must be at least 2D; got {shape!r}.")
    quanta = _coerce_scalar_or_map(
        basis.detected_quanta_per_pixel,
        expected_shape,
        field_name="detected_quanta_per_pixel",
    )
    visibility = _coerce_scalar_or_map(
        basis.visibility,
        expected_shape,
        field_name="visibility",
    )
    readout = np.asarray(basis.readout_variance_rad2, dtype=float)
    if readout.shape == ():
        readout_map = np.full(expected_shape, float(readout), dtype=float)
    else:
        if readout.shape != expected_shape:
            raise ValueError(
                "readout_variance_rad2 shape "
                f"{readout.shape} does not match phase frame shape {expected_shape}."
            )
        readout_map = readout
    if np.any(~np.isfinite(readout_map)) or np.any(readout_map < 0.0):
        raise ValueError("readout_variance_rad2 must be finite and non-negative.")

    shot = np.zeros(expected_shape, dtype=float)
    if bool(shot_noise_enabled):
        shot = 1.0 / (visibility * visibility * quanta)
    total = shot + (readout_map if bool(gaussian_noise_enabled) else 0.0)
    return np.maximum(total, float(variance_floor))


def _decode_complex_json_payload(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value.keys()) >= {"real", "imag"}:
        return complex(float(value["real"]), float(value["imag"]))
    if isinstance(value, (list, tuple)):
        return [_decode_complex_json_payload(item) for item in value]
    return value


def _complex_array_from_likelihood_payload(value: Any) -> np.ndarray:
    return np.asarray(_decode_complex_json_payload(value), dtype=np.complex128)


def analysis_noise_model_from_likelihood(
    payload: AnalysisNoiseModel | Mapping[str, Any],
    *,
    context: str = "analysis noise likelihood",
) -> AnalysisNoiseModel:
    """Return an :class:`AnalysisNoiseModel` from a serialized likelihood payload.

    The Fisher likelihood is a covariance model, not merely the diagonal
    variance image. Keeping this reconstruction in the shared noise contract
    prevents packet, fusion, axial, and SE(3) consumers from each inventing a
    different interpretation of row-correlated scan-line noise.
    """
    if isinstance(payload, AnalysisNoiseModel):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{context} must be an AnalysisNoiseModel or mapping; "
            f"got {type(payload).__name__}."
        )
    if "diagonal_variance" not in payload:
        raise ValueError(f"{context} is missing required field 'diagonal_variance'.")
    diagonal_variance = np.asarray(payload["diagonal_variance"], dtype=float)
    if diagonal_variance.size == 0:
        raise ValueError(f"{context} diagonal_variance is empty.")
    if np.any(~np.isfinite(diagonal_variance)) or np.any(diagonal_variance <= 0.0):
        raise ValueError(f"{context} diagonal_variance must be finite and positive.")
    row_couplings_raw = payload.get("row_correlated_couplings")
    row_couplings = None if row_couplings_raw is None else np.asarray(row_couplings_raw, dtype=float)
    if row_couplings is not None:
        if row_couplings.ndim != 3 or row_couplings.shape[1:] != diagonal_variance.shape:
            raise ValueError(
                f"{context} row_correlated_couplings must have shape "
                f"(components, H, W) matching diagonal_variance; got "
                f"{row_couplings.shape!r} and {diagonal_variance.shape!r}."
            )
        if np.any(~np.isfinite(row_couplings)):
            raise ValueError(f"{context} row_correlated_couplings must be finite.")
    component_variances = tuple(
        float(v) for v in payload.get("row_correlated_component_variances", ())
    )
    if row_couplings is not None and len(component_variances) != row_couplings.shape[0]:
        raise ValueError(
            f"{context} row_correlated_component_variances must contain one entry "
            "per coupling component."
        )
    if any((not np.isfinite(v)) or v < 0.0 for v in component_variances):
        raise ValueError(f"{context} row-correlated component variances must be finite and non-negative.")
    row_variance = float(payload.get("row_correlated_variance", 0.0))
    if not np.isfinite(row_variance) or row_variance < 0.0:
        raise ValueError(f"{context} row_correlated_variance must be finite and non-negative.")
    fourier_raw_raw = payload.get("fourier_sideband_raw_variance")
    fourier_mask_raw = payload.get("fourier_sideband_mask")
    fourier_phase_raw = payload.get("fourier_sideband_phase_correction")
    fourier_norm_raw = payload.get("fourier_sideband_output_normalization")
    fourier_raw = None if fourier_raw_raw is None else np.asarray(fourier_raw_raw, dtype=float)
    fourier_mask = None if fourier_mask_raw is None else np.asarray(fourier_mask_raw, dtype=bool)
    fourier_phase = (
        None
        if fourier_phase_raw is None
        else _complex_array_from_likelihood_payload(fourier_phase_raw)
    )
    fourier_norm = (
        None
        if fourier_norm_raw is None
        else _complex_array_from_likelihood_payload(fourier_norm_raw)
    )
    covariance_kind = str(payload.get("covariance_kind", "independent_pixels"))
    if covariance_kind == "fourier_sideband_demodulated_complex_field":
        if fourier_raw is None or fourier_mask is None:
            raise ValueError(
                f"{context} covariance_kind='fourier_sideband_demodulated_complex_field' "
                "requires fourier_sideband_raw_variance and fourier_sideband_mask."
            )
        if fourier_raw.shape != diagonal_variance.shape or fourier_mask.shape != diagonal_variance.shape:
            raise ValueError(
                f"{context} Fourier sideband raw variance/mask must match diagonal_variance "
                f"shape {diagonal_variance.shape}; got "
                f"{fourier_raw.shape} and {fourier_mask.shape}."
            )
        if np.any(~np.isfinite(fourier_raw)) or np.any(fourier_raw <= 0.0):
            raise ValueError(f"{context} fourier_sideband_raw_variance must be positive and finite.")
        if not np.any(fourier_mask):
            raise ValueError(f"{context} fourier_sideband_mask must select at least one coefficient.")
        if fourier_phase is not None:
            if fourier_phase.shape != diagonal_variance.shape:
                raise ValueError(
                    f"{context} fourier_sideband_phase_correction shape {fourier_phase.shape} "
                    f"must match diagonal_variance shape {diagonal_variance.shape}."
                )
            if np.any(~np.isfinite(fourier_phase.real)) or np.any(~np.isfinite(fourier_phase.imag)):
                raise ValueError(f"{context} fourier_sideband_phase_correction must be finite.")
            if not np.allclose(np.abs(fourier_phase), 1.0, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError(f"{context} fourier_sideband_phase_correction must be unit magnitude.")
        if fourier_norm is not None:
            if fourier_norm.shape != diagonal_variance.shape:
                raise ValueError(
                    f"{context} fourier_sideband_output_normalization shape "
                    f"{fourier_norm.shape} must match diagonal_variance shape "
                    f"{diagonal_variance.shape}."
                )
            if np.any(~np.isfinite(fourier_norm.real)) or np.any(~np.isfinite(fourier_norm.imag)):
                raise ValueError(f"{context} fourier_sideband_output_normalization must be finite.")
            if np.any(np.abs(fourier_norm) <= 0.0):
                raise ValueError(f"{context} fourier_sideband_output_normalization must be nonzero.")
    sideband_shift_raw = payload.get("fourier_sideband_shift", (0, 0))
    sideband_shift_tuple = tuple(int(v) for v in sideband_shift_raw)
    if len(sideband_shift_tuple) != 2:
        raise ValueError(f"{context} fourier_sideband_shift must be a (dy, dx) tuple.")
    return AnalysisNoiseModel(
        diagonal_variance=diagonal_variance,
        measurement_domain=str(payload.get("measurement_domain", "contrast")),
        signal_units=str(payload.get("signal_units", "contrast")),
        noise_variance_units=str(payload.get("noise_variance_units", "contrast_squared")),
        covariance_kind=covariance_kind,
        row_correlated_variance=row_variance,
        row_correlated_component_variances=component_variances,
        row_correlated_couplings=row_couplings,
        fourier_sideband_raw_variance=fourier_raw,
        fourier_sideband_mask=fourier_mask,
        fourier_sideband_shift=(int(sideband_shift_tuple[0]), int(sideband_shift_tuple[1])),
        fourier_sideband_phase_correction=fourier_phase,
        fourier_sideband_output_normalization=fourier_norm,
        fourier_sideband_output_conjugate=bool(payload.get("fourier_sideband_output_conjugate", False)),
        fourier_sideband_operator=str(payload.get("fourier_sideband_operator", "")),
        safe_for_ordering=bool(payload.get("safe_for_ordering", True)),
        safe_for_fusion=bool(payload.get("safe_for_fusion", True)),
        status_reason=str(payload.get("status_reason", "")),
    )

def analysis_noise_model_to_likelihood(
    noise_model: AnalysisNoiseModel | Mapping[str, Any],
    *,
    context: str = "analysis noise likelihood",
) -> dict[str, Any]:
    """Serialize a full analysis-noise likelihood without dropping covariance.

    The returned mapping intentionally includes both the diagonal projection and
    any structured covariance couplings. Candidate-keyed diagonal variance maps
    may be kept as report summaries, but this payload is the durable Fisher
    likelihood contract for saved packets and public comparison APIs.
    """
    model = analysis_noise_model_from_likelihood(noise_model, context=context)
    row_couplings = (
        None
        if model.row_correlated_couplings is None
        else np.asarray(model.row_correlated_couplings, dtype=float)
    )
    fourier_raw = (
        None
        if model.fourier_sideband_raw_variance is None
        else np.asarray(model.fourier_sideband_raw_variance, dtype=float)
    )
    fourier_mask = (
        None
        if model.fourier_sideband_mask is None
        else np.asarray(model.fourier_sideband_mask, dtype=bool)
    )
    fourier_phase = (
        None
        if model.fourier_sideband_phase_correction is None
        else np.asarray(model.fourier_sideband_phase_correction, dtype=np.complex128)
    )
    fourier_norm = (
        None
        if model.fourier_sideband_output_normalization is None
        else np.asarray(model.fourier_sideband_output_normalization, dtype=np.complex128)
    )
    return {
        "schema_version": ANALYSIS_NOISE_LIKELIHOOD_SCHEMA_VERSION,
        "diagonal_variance": np.asarray(model.diagonal_variance, dtype=float),
        "measurement_domain": str(model.measurement_domain),
        "signal_units": str(model.signal_units),
        "noise_variance_units": str(model.noise_variance_units),
        "covariance_kind": str(model.covariance_kind),
        "row_correlated_variance": float(model.row_correlated_variance),
        "row_correlated_component_variances": tuple(
            float(v) for v in model.row_correlated_component_variances
        ),
        "row_correlated_couplings": row_couplings,
        "fourier_sideband_raw_variance": fourier_raw,
        "fourier_sideband_mask": fourier_mask,
        "fourier_sideband_shift": tuple(int(v) for v in model.fourier_sideband_shift),
        "fourier_sideband_phase_correction": fourier_phase,
        "fourier_sideband_output_normalization": fourier_norm,
        "fourier_sideband_output_conjugate": bool(model.fourier_sideband_output_conjugate),
        "fourier_sideband_operator": str(model.fourier_sideband_operator),
        "safe_for_ordering": bool(model.safe_for_ordering),
        "safe_for_fusion": bool(model.safe_for_fusion),
        "status_reason": str(model.status_reason),
    }


__all__ = [
    "ANALYSIS_NOISE_LIKELIHOOD_SCHEMA_VERSION",
    "FISHER_LIKELIHOOD_ELIGIBILITY_CONTRACT_ID",
    "PHASE_LIKELIHOOD_SCHEMA_VERSION",
    "AnalysisNoiseModel",
    "AnalysisNoiseSummary",
    "INDEPENDENT_PIXEL_NOISE_CONTRACT_ID",
    "IndependentPixelNoiseModel",
    "FisherLikelihoodEligibility",
    "PhaseLikelihoodBasis",
    "phase_variance_from_likelihood_basis",
    "scale_analysis_noise_model",
    "fourier_sideband_demodulated_noise_model",
    "independent_pixel_noise_model",
    "fisher_noise_input_to_analysis_model",
    "analysis_noise_model_from_likelihood",
    "analysis_noise_model_to_likelihood",
    "summarize_analysis_noise_model",
    "resolve_fisher_likelihood_eligibility",
]
