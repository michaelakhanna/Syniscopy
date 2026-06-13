from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def test_analytic_polarizability_vectorial_debye_transports_dipole_not_componentwise_multiplier() -> None:
    from config import default_params
    from optical_scattering import analytic_polarizability_dipole_vector, optical_scattering_render_multiplier
    from optics import compute_complex_psf_stack
    from particle_specs import ParticleComponentSpec

    params = default_params()
    params.update(
        {
            "imaging_model": "bright_field",
            "optical_field_backend": "vectorial_debye",
            "optical_scattering_model": "analytic_polarizability",
            "vectorial_detection_mode": "full_vector",
            "polarization_model": "linear_x",
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
        }
    )
    refractive_index = 1.6 + 0.02j
    component = ParticleComponentSpec(
        shape="ellipsoid",
        offset_nm=(0.0, 0.0, 0.0),
        diameter_nm=120.0,
        axes_nm=(120.0, 30.0, 30.0),
        refractive_index=refractive_index,
    )
    interp = compute_complex_psf_stack(
        params,
        120.0,
        refractive_index,
        [0.0],
        optical_scattering_model="analytic_polarizability",
        component_geometry=component,
    )
    field = interp.field_at([0.0], orientation_matrix=np.eye(3))[0]
    dipole = analytic_polarizability_dipole_vector(
        params,
        component_geometry=component,
        material_properties=None,
        orientation_matrix=np.eye(3),
        fallback_refractive_index=refractive_index,
    )
    basis = interp._basis_at_single(0.0)
    expected = np.tensordot(dipole, basis, axes=(0, 0))
    old_componentwise = dipole[:, None, None] * basis[0]

    assert interp.metadata["analytic_polarizability_vectorial_transport"] == "debye_operator_applied_to_dipole"
    assert optical_scattering_render_multiplier(
        params,
        component_geometry=component,
        material_properties=None,
        orientation_matrix=np.eye(3),
        field_metadata=interp.metadata,
    ) == 1.0 + 0.0j
    assert field.shape == (3, 16, 16)
    assert np.allclose(field, expected)
    assert np.linalg.norm(field[1]) > 1.0e-3 * max(np.linalg.norm(field[0]), 1.0e-30)
    assert np.linalg.norm(field - old_componentwise) > 1.0e-3 * max(np.linalg.norm(field), 1.0e-30)


def test_dpc_asymmetric_illumination_half_disc_boundary_is_half_weighted() -> None:
    from config import default_params
    from imaging_models.dpc import DifferentialPhaseContrastImagingModel

    params = default_params()
    params.update(
        {
            "dpc_transfer_model": "asymmetric_illumination",
            "dpc_source_samples": 19,
            "dpc_illumination_sigma": 0.7,
        }
    )

    for axis, axis_index, expected_boundary_count in (
        ("x", 0, 3),
        ("y", 1, 5),
    ):
        pos_pts, pos_weights = DifferentialPhaseContrastImagingModel._half_disc_source_quadrature(
            params,
            axis=axis,
            sign=1.0,
        )
        neg_pts, neg_weights = DifferentialPhaseContrastImagingModel._half_disc_source_quadrature(
            params,
            axis=axis,
            sign=-1.0,
        )
        pos_boundary = np.isclose(pos_pts[:, axis_index], 0.0)
        neg_boundary = np.isclose(neg_pts[:, axis_index], 0.0)

        assert int(np.count_nonzero(pos_boundary)) == expected_boundary_count
        assert int(np.count_nonzero(neg_boundary)) == expected_boundary_count
        assert np.allclose(pos_weights[pos_boundary], 0.5)
        assert np.allclose(neg_weights[neg_boundary], 0.5)
        assert np.all(pos_weights[~pos_boundary] == 1.0)
        assert np.all(neg_weights[~neg_boundary] == 1.0)
        assert np.isclose(float(np.sum(pos_weights) + np.sum(neg_weights)), 19.0)
