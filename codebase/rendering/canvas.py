"""Render-canvas sizing and guard-band helpers."""

from __future__ import annotations

import numpy as np

from config import SamplingGeometry
from imaging_models import get_imaging_model
from particle_model import ParticleInstance, ParticleType
from particle_specs import get_particle_specs

from .airy_support import estimate_optical_filter_guard_radius_pixels, estimate_psf_padding_radius_pixels

def _source_footprint_guard_radius_pixels(
    params: dict,
    particle_instances: list[ParticleInstance] | None,
) -> int:
    """Return a source-map guard radius in oversampled canvas pixels."""
    sampling = SamplingGeometry.from_params(params)
    max_diameter_nm = 0.0
    if particle_instances is None:
        for spec in get_particle_specs(params):
            for component in spec.components:
                offset = np.asarray(component.offset_nm, dtype=float)
                offset_radius_nm = float(np.linalg.norm(offset)) if offset.size >= 2 else 0.0
                sub_radius_nm = float(component.bounding_radius_nm)
                max_diameter_nm = max(max_diameter_nm, 2.0 * (offset_radius_nm + sub_radius_nm))
    else:
        for instance in particle_instances:
            ptype: ParticleType = instance.particle_type
            max_diameter_nm = max(max_diameter_nm, float(getattr(ptype, "diameter_nm", 0.0)))
            if getattr(ptype, "is_composite", False) and getattr(ptype, "sub_particles", None):
                for sub in ptype.sub_particles:
                    offset = np.asarray(getattr(sub, "offset_nm", [0.0, 0.0, 0.0]), dtype=float)
                    offset_radius_nm = float(np.linalg.norm(offset)) if offset.size >= 2 else 0.0
                    geometry = getattr(sub, "component_geometry", None)
                    sub_radius_nm = (
                        float(geometry.bounding_radius_nm)
                        if geometry is not None
                        else 0.5 * float(getattr(sub, "diameter_nm", 0.0))
                    )
                    max_diameter_nm = max(max_diameter_nm, 2.0 * (offset_radius_nm + sub_radius_nm))
    guard = (
        0.5
        * max_diameter_nm
        / sampling.detector_pixel_size_nm
        * float(sampling.psf_oversampling_factor)
    )
    return int(np.ceil(max(guard, 0.0))) + 2


def resolve_render_canvas_geometry(
    params: dict,
    particle_instances: list[ParticleInstance] | None = None,
    imaging_model=None,
) -> dict[str, int | float | str]:
    """Resolve render canvas, guard-band, and crop geometry for one model."""
    sampling = SamplingGeometry.from_params(params)
    img_size = sampling.image_size_pixels
    pixel_size_nm = sampling.detector_pixel_size_nm
    os_factor = sampling.psf_oversampling_factor
    os_size = sampling.model_canvas_shape[0]
    model = imaging_model if imaging_model is not None else get_imaging_model(params)
    requires_optical_scattered_field = bool(
        getattr(model, "requires_optical_scattered_field", True)
    )
    uses_particle_sources = bool(
        getattr(model, "uses_particle_material_sources", False)
    )
    guard_sources: list[str] = []
    render_guard_radius = 0
    if requires_optical_scattered_field:
        render_guard_radius = estimate_psf_padding_radius_pixels(params)
        guard_sources.append("optical_psf_padding")
    if uses_particle_sources and particle_instances is not None:
        source_guard = _source_footprint_guard_radius_pixels(params, particle_instances)
        if source_guard > render_guard_radius:
            render_guard_radius = source_guard
        guard_sources.append("source_footprint")
    if bool(getattr(model, "requires_pre_crop_optical_filtering", False)):
        model_guard = model.filter_guard_radius_pixels(params)
        if model_guard is None and requires_optical_scattered_field:
            model_guard = estimate_optical_filter_guard_radius_pixels(params)
            guard_sources.append("optical_filter")
        elif model_guard is not None:
            guard_sources.append("model_filter")
        if model_guard is not None:
            render_guard_radius = max(render_guard_radius, int(np.ceil(float(model_guard))))
    render_guard_radius = int(max(render_guard_radius, 0))
    os_canvas_size = os_size + 2 * render_guard_radius
    crop_start = render_guard_radius
    crop_end = crop_start + os_size
    return {
        "detector_image_size_pixels": int(img_size),
        "detector_pixel_size_nm": float(pixel_size_nm),
        "psf_oversampling_factor": int(os_factor),
        "os_size_pixels": int(os_size),
        "render_guard_radius_pixels": int(render_guard_radius),
        "os_canvas_size_pixels": int(os_canvas_size),
        "model_canvas_size_pixels": int(os_canvas_size),
        "model_canvas_pixel_size_nm": float(pixel_size_nm) / float(os_factor),
        "crop_start": int(crop_start),
        "crop_end": int(crop_end),
        "layout_extent_nm": float(os_canvas_size) * float(pixel_size_nm) / float(os_factor),
        "guard_source": "+".join(guard_sources) if guard_sources else "none",
        "requires_optical_scattered_field": bool(requires_optical_scattered_field),
        "uses_particle_material_sources": bool(uses_particle_sources),
    }
