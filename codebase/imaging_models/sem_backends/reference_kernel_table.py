"""ReferenceKernelSEMBackend SEM backend."""

from __future__ import annotations

from config.runtime import SemSettings
from imaging_models.sem_source import SEMMaterialSourceCanvas

from ._metadata import (
    Any,
    Mapping,
    Path,
    SEMTransportBackendError,
    SEM_REFERENCE_KERNEL_SCHEMA_VERSION,
    _detector_takeoff_acceptance_gain,
    _electrons_from_beam_current,
    _finite_nonnegative,
    _gaussian_blur,
    _gradient_components,
    _sha256_file,
    _validate_takeoff_angle_deg,
    attach_backend_fidelity_metadata,
    json,
    np,
)
from .reference_kernel_examples import example_sem_reference_kernel_payload

class ReferenceKernelSEMBackend:
    """SEM backend driven by a user supplied reference-kernel table.

    The backend interpolates yield from a JSON table over key operating
    dimensions, without depending on external simulators. The table is expected
    to contain deterministic entries for material/energy/geometry/depth/angle and
    source-level yield.
    """

    backend_mode = "reference_kernel_table"

    def __init__(
        self,
        params: dict,
        *,
        canvas_pitch_nm: float,
        probe_sigma_px: float,
    ) -> None:
        self.canvas_pitch_nm = _finite_nonnegative("canvas_pitch_nm", canvas_pitch_nm, minimum=1e-12)
        self.probe_sigma_px = _finite_nonnegative("sem_probe_sigma_px", probe_sigma_px, minimum=0.0)
        self.backend_mode = self.__class__.backend_mode
        sem_settings = SemSettings.from_params(params)

        raw_path = sem_settings.reference_kernel_path
        if not raw_path:
            raise SEMTransportBackendError(
                "parameters['sem_backend']='reference_kernel_table' requires "
                "parameters['sem_reference_kernel_path']."
            )
        self.path = Path(str(raw_path)).expanduser()
        if not self.path.exists():
            raise SEMTransportBackendError(f"SEM reference-kernel table not found: {self.path}")
        self.sha256 = _sha256_file(self.path)

        expected = sem_settings.reference_kernel_sha256
        if expected and str(expected).lower() != self.sha256.lower():
            raise SEMTransportBackendError(
                "SEM reference-kernel checksum mismatch: "
                f"expected {expected}, observed {self.sha256}."
            )

        try:
            payload = json.loads(self.path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SEMTransportBackendError(
                f"Could not parse SEM reference-kernel JSON {self.path!s}: {exc}"
            ) from exc
        self.payload = self._validate_payload(payload)
        self.reference_status = str(
            self.payload.get("validation_status", "physics_based_unvalidated")
        ).strip().lower()
        if self.reference_status == "reference_validated" and not expected:
            raise SEMTransportBackendError(
                "SEM reference-kernel table declares reference_validated, but "
                "parameters['sem_reference_kernel_sha256'] was not supplied."
            )
        self.validation_status = (
            "external_artifact_required"
            if self.reference_status == "reference_validated"
            else "diagnostic_only"
        )

        self._acceleration_kV = sem_settings.acceleration_kV
        self._baseline = sem_settings.baseline_yield
        self._detector_acceptance = sem_settings.detector_acceptance
        self._topography_gain = sem_settings.topography_contrast_gain
        self._takeoff_angle_deg = sem_settings.detector_takeoff_angle_deg
        _validate_takeoff_angle_deg(self._takeoff_angle_deg)
        self._source_depth_nm = sem_settings.reference_source_depth_nm
        self._material_name = sem_settings.reference_material
        self._geometry_name = sem_settings.reference_geometry
        self._beam_current_nA = sem_settings.beam_current_nA
        self._dwell_time_us = sem_settings.dwell_time_us
        self._electrons_per_pixel_reference = sem_settings.electrons_per_pixel
        self._incident_angle_deg = sem_settings.reference_incident_angle_deg
        self._detector_direction_xy = np.asarray(sem_settings.detector_direction_xy, dtype=float)

        rows = self.payload.get("kernel_rows")
        if rows is None:
            rows = self.payload.get("reference_rows")
        if rows is None:
            raise SEMTransportBackendError(
                "SEM reference-kernel table must contain a non-empty list under '\"kernel_rows\"' or '\"reference_rows\"'."
            )
        if not isinstance(rows, list) or len(rows) == 0:
            raise SEMTransportBackendError("SEM reference-kernel table must provide one or more rows.")
        self._rows = self._normalize_rows(rows)
        if len(self._rows) == 0:
            raise SEMTransportBackendError("SEM reference-kernel table contained no valid rows.")
        self._axis_scales = self._axis_scales_from_rows(self._rows)

    def _electrons_from_beam_current(self) -> float | None:
        return _electrons_from_beam_current(self._beam_current_nA, self._dwell_time_us)

    def electrons_per_pixel(self) -> float:
        return self._electrons_from_beam_current() or self._electrons_per_pixel_reference

    @staticmethod
    def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        schema = str(payload.get("schema_version", "")).strip()
        if schema != SEM_REFERENCE_KERNEL_SCHEMA_VERSION:
            raise SEMTransportBackendError(
                f"SEM reference-kernel schema mismatch: expected {SEM_REFERENCE_KERNEL_SCHEMA_VERSION!r}; got {schema!r}."
            )
        return dict(payload)

    @staticmethod
    def _normalize_rows(rows: list[Any]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        required = {
            "source",
            "yield",
            "beam_energy_kV",
            "source_depth_nm",
            "takeoff_angle_deg",
            "geometry",
            "material",
        }
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise SEMTransportBackendError(
                    f"SEM reference-kernel row {idx} must be a mapping/dictionary."
                )
            missing = [k for k in required if k not in row]
            if missing:
                raise SEMTransportBackendError(
                    f"SEM reference-kernel row {idx} missing keys: {missing}."
                )
            try:
                normalized = {
                    "source": float(row["source"]),
                    "yield": float(row["yield"]),
                    "beam_energy_kV": float(row["beam_energy_kV"]),
                    "source_depth_nm": float(row["source_depth_nm"]),
                    "takeoff_angle_deg": float(row["takeoff_angle_deg"]),
                    "incident_angle_deg": float(row.get("incident_angle_deg", 0.0)),
                    "geometry": str(row["geometry"]).strip().lower() or "normal",
                    "material": str(row["material"]).strip().lower() or "default",
                    "backscatter_yield": float(row.get("backscatter_yield", 0.0)),
                }
            except (TypeError, ValueError) as exc:
                raise SEMTransportBackendError(f"SEM reference-kernel row {idx} has invalid numeric value: {exc}") from exc
            normalized["yield"] = max(normalized["yield"], 0.0)
            if normalized["source_depth_nm"] < 0.0:
                normalized["source_depth_nm"] = 0.0
            out.append(normalized)
        return out

    @staticmethod
    def _axis_scales_from_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        scales: dict[str, float] = {}
        for key in ("beam_energy_kV", "source_depth_nm", "incident_angle_deg", "takeoff_angle_deg", "source"):
            values = np.array([float(r[key]) for r in rows], dtype=float)
            if values.size == 0:
                scales[key] = 1.0
                continue
            span = float(values.max() - values.min())
            if not np.isfinite(span) or span <= 0.0:
                scales[key] = 1.0
            else:
                scales[key] = span
        return scales

    def _select_rows(self, material: str, geometry: str) -> list[dict[str, float]]:
        exact_candidates = []
        wildcard_candidates = []
        default_candidates = []
        for row in self._rows:
            row_material = row["material"]
            if row_material == material:
                material_rank = "exact"
            elif row_material in {"*", "all"}:
                material_rank = "wildcard"
            elif row_material == "default":
                material_rank = "default"
            else:
                continue
            row_geometry = row["geometry"]
            if row_geometry == geometry:
                geometry_rank = "exact"
            elif row_geometry in {"*", "all"}:
                geometry_rank = "wildcard"
            elif row_geometry == "default":
                geometry_rank = "default"
            else:
                continue
            if material_rank == "exact" and geometry_rank == "exact":
                exact_candidates.append(row)
            elif material_rank != "default" and geometry_rank != "default":
                wildcard_candidates.append(row)
            else:
                default_candidates.append(row)

        key_candidates = exact_candidates or wildcard_candidates or default_candidates

        if not key_candidates:
            raise SEMTransportBackendError(
                "SEM reference-kernel table has no rows for "
                f"material={material!r}, geometry={geometry!r}."
            )

        return key_candidates

    def _detector_gain(self) -> float:
        return _detector_takeoff_acceptance_gain(
            self._detector_acceptance,
            self._takeoff_angle_deg,
        )

    def _interpolate_rows(
        self,
        source: np.ndarray,
        *,
        material: str | None = None,
        geometry: str | None = None,
    ) -> np.ndarray:
        material_name = str(material or self._material_name).strip().lower() or "default"
        geometry_name = str(geometry or self._geometry_name).strip().lower() or "normal"
        rows = self._select_rows(material_name, geometry_name)
        if len(rows) == 0:
            return np.zeros_like(source, dtype=float)

        source_query = np.asarray(source, dtype=float)
        if not np.all(np.isfinite(source_query)):
            raise FloatingPointError("SEM reference-kernel source contains non-finite values.")

        q = {
            "beam_energy_kV": self._acceleration_kV,
            "source_depth_nm": self._source_depth_nm,
            "incident_angle_deg": self._incident_angle_deg,
            "takeoff_angle_deg": self._takeoff_angle_deg,
        }

        num = np.zeros_like(source_query, dtype=float)
        den = np.zeros_like(source_query, dtype=float)
        for row in rows:
            d = 0.0
            d += abs(q["beam_energy_kV"] - row["beam_energy_kV"]) / self._axis_scales["beam_energy_kV"]
            d += abs(q["source_depth_nm"] - row["source_depth_nm"]) / self._axis_scales["source_depth_nm"]
            d += abs(q["incident_angle_deg"] - row["incident_angle_deg"]) / self._axis_scales["incident_angle_deg"]
            d += abs(q["takeoff_angle_deg"] - row["takeoff_angle_deg"]) / self._axis_scales["takeoff_angle_deg"]
            row_yield = row["yield"]
            d_with_source = d + np.abs(source_query - row["source"]) / self._axis_scales["source"]
            weight = 1.0 / (1.0 + d_with_source)
            num += weight * row_yield
            den += weight
        den = np.where(den <= 0.0, 1.0, den)
        out = num / den
        return np.maximum(out, 0.0)

    def _topography_term(self, source: np.ndarray) -> np.ndarray:
        if self._topography_gain <= 0.0:
            return np.zeros_like(source)
        gx, gy = _gradient_components(source, self.canvas_pitch_nm)
        directed = self._detector_direction_xy[0] * gx + self._detector_direction_xy[1] * gy
        topo = np.maximum(directed, 0.0)
        return self._topography_gain * _gaussian_blur(topo, max(self.probe_sigma_px * 0.5, 0.0))

    @staticmethod
    def _project_source_for_table(source: np.ndarray) -> np.ndarray:
        source_query = np.asarray(source, dtype=float)
        if source_query.ndim == 3:
            source_query = np.sum(source_query, axis=0)
        return source_query

    def yield_from_source(self, source: np.ndarray, *, baseline: float = 0.0) -> np.ndarray:
        if isinstance(source, SEMMaterialSourceCanvas):
            output = np.zeros(source.shape[-2:], dtype=float)
            for key, channel_source in source.channels.items():
                channel_query = self._project_source_for_table(channel_source)
                interpolated = self._interpolate_rows(
                    key.se_yield_coefficient * channel_query,
                    material=key.material_name,
                    geometry=self._geometry_name,
                )
                if self._topography_gain > 0.0:
                    interpolated = interpolated + self._topography_term(channel_query)
                output += interpolated
            output = np.maximum(float(baseline) + self._detector_gain() * output, 0.0)
            output = _gaussian_blur(output, max(self.probe_sigma_px, 0.0))
            if not np.all(np.isfinite(output)):
                raise FloatingPointError("SEM reference-kernel backend produced non-finite yield map.")
            return output

        source_query = self._project_source_for_table(source)
        interpolated = self._interpolate_rows(source_query)
        if self._topography_gain > 0.0:
            interpolated = interpolated + self._topography_term(source_query)
        output = np.maximum(baseline + self._detector_gain() * interpolated, 0.0)
        output = _gaussian_blur(output, max(self.probe_sigma_px, 0.0))
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("SEM reference-kernel backend produced non-finite yield map.")
        return output

    def contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        if isinstance(source, SEMMaterialSourceCanvas):
            zero = SEMMaterialSourceCanvas(
                shape=source.shape,
                channels={key: np.zeros_like(value, dtype=float) for key, value in source.channels.items()},
            )
            return self.yield_from_source(source, baseline=0.0) - self.yield_from_source(
                zero,
                baseline=0.0,
            )
        src = self._project_source_for_table(source)
        return self.yield_from_source(src, baseline=0.0) - self.yield_from_source(
            np.zeros_like(src, dtype=float),
            baseline=0.0,
        )

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        raw = params or {}
        fidelity = "high_fidelity" if self.reference_status == "reference_validated" else "physics_based"
        meta = {
            "kind": "sem_reference_kernel",
            "backend_mode": self.backend_mode,
            "backend_fidelity_level": fidelity,
            "sem_backend": self.backend_mode,
            "backend_name": self.backend_mode,
            "equations_or_model_family": "interpolated_reference_kernel_sem_transport",
            "implemented_approximation_level": fidelity,
            "native_operating_assumptions": "table-lookup interpolation over material/energy/geometry/depth/angle",
            "comparison_contract_id": str(raw.get("comparison_contract_id", "Contract-NR")),
            "artifact_provenance_id": raw.get("artifact_provenance_id", None),
            "fidelity_label": (
                "sem_reference_kernel_external_artifact_required"
                if self.reference_status == "reference_validated"
                else "sem_reference_kernel_diagnostic_only"
            ),
            "validation_status": self.validation_status,
            "forward_observable": "interpolated SEM secondary-electron yield table",
            "acceleration_kV": self._acceleration_kV,
            "detector_takeoff_angle_deg": self._takeoff_angle_deg,
            "source_depth_nm": self._source_depth_nm,
            "reference_kernel_path": str(self.path),
            "sem_reference_kernel_sha256": self.sha256,
            "material": self._material_name,
            "geometry": self._geometry_name,
            "reference_backend_metadata": {
                "schema_version": SEM_REFERENCE_KERNEL_SCHEMA_VERSION,
                "path": str(self.path),
                "sha256": self.sha256,
                "row_count": len(self._rows),
                "reference_status": self.reference_status,
                "validation_status": self.validation_status,
                "claim_maturity_gate": self.validation_status,
            },
            "electrons_per_pixel": self.electrons_per_pixel(),
            "baseline_yield": self._baseline,
            "detector_acceptance": self._detector_acceptance,
        }
        return attach_backend_fidelity_metadata(
            meta,
            params=raw,
            backend_name=self.backend_mode,
            equations_or_model_family="interpolated_reference_kernel_sem_transport",
            implemented_approximation_level=fidelity,
            native_operating_assumptions=meta["native_operating_assumptions"],
            comparison_contract_id=str(raw.get("comparison_contract_id", "Contract-NR")),
            artifact_provenance_id=raw.get("artifact_provenance_id", None),
        )


def write_example_sem_reference_kernel(path: str | Path) -> Path:
    """Write a minimal unvalidated reference-kernel table."""
    target = Path(path)
    payload = example_sem_reference_kernel_payload()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = ["ReferenceKernelSEMBackend", "write_example_sem_reference_kernel"]
