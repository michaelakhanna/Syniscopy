from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def test_static_fusion_keeps_independent_identical_fisher_channels() -> None:
    from lab_fisher_report.tables import _select_fusion_fisher_inputs

    fisher = np.array([[4.0, 0.0], [0.0, 9.0]])
    selected, excluded = _select_fusion_fisher_inputs(
        {
            "per_microscope": {
                "scope_a": {
                    "safe_for_fusion": True,
                    "fisher_matrix": fisher,
                    "singular": False,
                },
                "scope_b": {
                    "safe_for_fusion": True,
                    "fisher_matrix": fisher.copy(),
                    "singular": False,
                },
            }
        }
    )

    assert set(selected) == {"scope_a", "scope_b"}
    assert excluded == {}


def test_lab_fisher_configured_profile_defaults_match_paper_contract() -> None:
    from config.runtime import OpticalInstrumentSettings, SamplingGeometry
    from lab_fisher_report.cli import TEMPLATE_OVERRIDES
    from lab_fisher_report.params_assembly import _make_template_base_params
    from lab_fisher_report.report_contracts import (
        REPORT_CONFIGURED_PROFILE_DEFAULTS,
        REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS,
        assert_report_configured_profile_defaults,
        assert_report_configured_profile_particle_defaults,
    )
    from material_optical_catalog import lookup_refractive_index

    assert TEMPLATE_OVERRIDES["image_size_pixels"] == 192
    assert TEMPLATE_OVERRIDES["pixel_size_nm"] == 65.0
    assert TEMPLATE_OVERRIDES["pupil_samples"] == 384
    assert TEMPLATE_OVERRIDES["psf_oversampling_factor"] == 2
    assert {
        key: TEMPLATE_OVERRIDES[key]
        for key in REPORT_CONFIGURED_PROFILE_DEFAULTS
    } == REPORT_CONFIGURED_PROFILE_DEFAULTS
    template_particle = TEMPLATE_OVERRIDES["particles"][0]
    template_component = template_particle["components"][0]
    assert len(TEMPLATE_OVERRIDES["particles"]) == 1
    assert len(template_particle["components"]) == 1
    assert template_component["shape"] == REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS["shape"]
    assert template_component["diameter_nm"] == REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS["diameter_nm"]
    assert template_particle["motion"]["hydrodynamic_diameter_nm"] == REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS[
        "hydrodynamic_diameter_nm"
    ]
    assert template_component["material"] == REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS["material"]
    assert lookup_refractive_index(template_component["material"], 532.0).imag > 0.0

    params = _make_template_base_params()
    assert_report_configured_profile_defaults(params)
    assert_report_configured_profile_particle_defaults(params)
    sampling = SamplingGeometry.from_params(params)
    instrument = OpticalInstrumentSettings.from_params(params)

    assert sampling.image_size_pixels == 192
    assert sampling.detector_pixel_size_nm == 65.0
    assert sampling.psf_oversampling_factor == 2
    assert instrument.pupil_samples == 384
    assert instrument.vectorial_pupil_samples == 384
    assert instrument.vectorial_pupil_samples_is_explicit is False


def test_dynamic_fusion_does_not_double_count_shared_prior() -> None:
    from config import default_params
    from experiment_contracts import ConvergenceStatus
    from lab_fisher_report.tables import _dynamic_fusion_rows_from_fisher_sequences

    zero_sequence = [np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=float)]
    bright_params = default_params()
    bright_params["imaging_model"] = "bright_field"
    dark_params = default_params()
    dark_params["imaging_model"] = "dark_field"
    dynamic_summary = {
        "dynamic_enabled": True,
        "state_axes": ["x", "y"],
        "state_transition_matrix": np.eye(2).tolist(),
        "process_noise_covariance": (np.eye(2) * 10.0).tolist(),
        "initial_covariance": (np.eye(2) * 100.0).tolist(),
        "fps": 5.0,
        "state_transition_fps": 5.0,
        "process_noise_fps": 5.0,
        "measurement_domain": "contrast",
        "signal_units": "contrast",
        "noise_variance_units": "contrast^2",
        "state_axis_units": {"x": "nm", "y": "nm"},
        "process_model": "brownian_translation_lateral_xy",
    }
    ranking_summary = {
        "sequence_crlb_model": "dynamic_bayesian_estimator",
        "safe_for_ordering": True,
        "safe_for_fusion": False,
        "convergence_status": ConvergenceStatus.FINITE_CONVERGED.value,
        "status_reason": "ranking uses dynamic posterior; fusion uses joint sequence",
        "modality": "bright_field",
    }

    rows, excluded, eligible = _dynamic_fusion_rows_from_fisher_sequences(
        {
            "scope_a": zero_sequence,
            "scope_b": [matrix.copy() for matrix in zero_sequence],
        },
        {
            "scope_a": dict(dynamic_summary, microscope="scope_a", modality="bright_field"),
            "scope_b": dict(dynamic_summary, microscope="scope_b", modality="dark_field"),
        },
        {
            "scope_a": dict(ranking_summary, microscope="scope_a", modality="bright_field"),
            "scope_b": dict(ranking_summary, microscope="scope_b", modality="dark_field"),
        },
        max_k=2,
        include_full=False,
        microscope_profile_cards={
            "scope_a": bright_params,
            "scope_b": dark_params,
        },
    )

    assert excluded == {}
    assert set(eligible) == {"scope_a", "scope_b"}
    by_size = {int(row["subset_size"]): row for row in rows}
    assert set(by_size) == {1, 2}
    assert np.isclose(
        float(by_size[2]["fusion_sigma_xy_nm"]),
        float(by_size[1]["fusion_sigma_xy_nm"]),
        rtol=1e-12,
        atol=0.0,
    )
    assert np.isclose(float(by_size[2]["fusion_gain_xy"]), 1.0, rtol=1e-12, atol=0.0)


def test_shared_latent_scene_views_match_at_common_physical_times() -> None:
    from config import default_params
    from lab_fisher_report.shared_latent_scene import build_shared_latent_scene

    scene_params = default_params()
    scene_params["random_seed"] = 123
    scope_a = dict(scene_params)
    scope_a["fps"] = 10.0
    scope_a["num_frames"] = 2
    scope_b = dict(scene_params)
    scope_b["fps"] = 30.0
    scope_b["num_frames"] = 5

    shared = build_shared_latent_scene(
        scene_params,
        {
            "scope_a": scope_a,
            "scope_b": scope_b,
        },
    )

    assert shared.schedule.fusion_time_alignment == "asynchronous"
    view_a = shared.view_for_microscope("scope_a")
    view_b = shared.view_for_microscope("scope_b")
    assert view_a["latent_scene_id"] == shared.provenance_id
    assert view_b["latent_scene_id"] == shared.provenance_id
    assert np.allclose(view_a["latent_times_s"], [0.05, 0.15], rtol=0.0, atol=1.0e-12)
    assert np.allclose(
        view_a["trajectories_nm"][:, 0, :],
        view_b["trajectories_nm"][:, 1, :],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        view_a["trajectories_nm"][:, 1, :],
        view_b["trajectories_nm"][:, 4, :],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_asynchronous_shared_latent_schedule_is_not_static_same_state_rankable() -> None:
    from lab_fisher_report.tables import _build_sequence_summary_rows

    frame = {
        "fisher_matrix": np.eye(2, dtype=float),
        "safe_for_ordering": True,
        "safe_for_fusion": True,
        "fusion_time_alignment": "asynchronous",
        "same_latent_scene": True,
        "latent_scene_id": "latent-scene-test",
        "latent_schedule_id": "latent-schedule-test",
        "state_time_policy": "union_frame_center_grid",
        "shared_coordinate_frame": "lab_report_shared_scene_xy_nm",
    }
    rows, summary = _build_sequence_summary_rows(
        "bright_field",
        [frame],
        microscope="scope_a",
    )

    assert summary["same_state_assumption"] is False
    assert summary["safe_for_ordering"] is False
    assert summary["safe_for_fusion"] is False
    assert summary["fusion_time_alignment"] == "asynchronous"
    assert rows[0]["same_state_assumption"] is False
    assert rows[0]["latent_scene_id"] == "latent-scene-test"


def test_dynamic_fusion_frame_count_compatibility_is_subset_local() -> None:
    from config import default_params
    from experiment_contracts import ConvergenceStatus
    from lab_fisher_report.tables import _dynamic_fusion_rows_from_fisher_sequences

    def sequence(value: float, frames: int) -> list[np.ndarray]:
        return [np.eye(2, dtype=float) * float(value) for _ in range(frames)]

    params_by_scope = {}
    for name, modality in (
        ("scope_a", "bright_field"),
        ("scope_b", "dark_field"),
        ("scope_c", "coherent_bright_field"),
    ):
        params = default_params()
        params["imaging_model"] = modality
        params_by_scope[name] = params

    dynamic_summary = {
        "dynamic_enabled": True,
        "state_axes": ["x", "y"],
        "state_transition_matrix": np.eye(2).tolist(),
        "process_noise_covariance": np.zeros((2, 2), dtype=float).tolist(),
        "initial_covariance": (np.eye(2) * 10.0).tolist(),
        "fps": 5.0,
        "state_transition_fps": 5.0,
        "process_noise_fps": 5.0,
        "measurement_domain": "contrast",
        "signal_units": "contrast",
        "noise_variance_units": "contrast^2",
        "state_axis_units": {"x": "nm", "y": "nm"},
        "process_model": "brownian_translation_lateral_xy",
    }
    ranking_summary = {
        "sequence_crlb_model": "dynamic_bayesian_estimator",
        "safe_for_ordering": True,
        "safe_for_fusion": False,
        "convergence_status": ConvergenceStatus.FINITE_CONVERGED.value,
        "status_reason": "ranking uses dynamic posterior; fusion uses joint sequence",
    }

    rows, excluded, eligible = _dynamic_fusion_rows_from_fisher_sequences(
        {
            "scope_a": sequence(1.0, 3),
            "scope_b": sequence(2.0, 2),
            "scope_c": sequence(3.0, 2),
        },
        {
            "scope_a": dict(dynamic_summary, microscope="scope_a", modality="bright_field"),
            "scope_b": dict(dynamic_summary, microscope="scope_b", modality="dark_field"),
            "scope_c": dict(dynamic_summary, microscope="scope_c", modality="coherent_bright_field"),
        },
        {
            "scope_a": dict(ranking_summary, microscope="scope_a", modality="bright_field"),
            "scope_b": dict(ranking_summary, microscope="scope_b", modality="dark_field"),
            "scope_c": dict(ranking_summary, microscope="scope_c", modality="coherent_bright_field"),
        },
        max_k=2,
        include_full=False,
        microscope_profile_cards=params_by_scope,
    )

    assert excluded == {}
    assert set(eligible) == {"scope_a", "scope_b", "scope_c"}
    by_size = {int(row["subset_size"]): row for row in rows}
    assert by_size[2]["microscopes_used"] == "scope_b;scope_c"
    assert int(by_size[2]["fusion_frame_count"]) == 2
    assert np.isclose(
        float(by_size[2]["fusion_sigma_xy_nm"]),
        np.sqrt(2.0 / 10.1),
        rtol=1e-12,
        atol=0.0,
    )


def test_dynamic_posterior_summary_is_rankable_but_not_static_fusion_input() -> None:
    from lab_fisher_report.tables import _dynamic_sequence_summary_to_ranking_summary

    dynamic_summary = {
        "dynamic_enabled": True,
        "dynamic_covariance_matrices": [
            [[25.0, 0.0], [0.0, 36.0]],
        ],
    }
    static_summary = {
        "microscope": "scope_a",
        "modality": "bright_field",
        "num_frames": 2,
        "measurement_domain": "contrast",
        "signal_units": "contrast",
        "noise_variance_units": "contrast^2",
        "safe_for_linear_fisher_variance": True,
        "safe_for_covariance_fisher_variance": True,
        "detector_safe_for_report_fisher": True,
        "fisher_likelihood_uses_covariance": False,
        "fisher_variance_model_scope": "linearized_contrast",
        "covariance_fisher_variance_model_scope": "",
        "detector_likelihood_status": "linearized_fisher_safe",
        "derivative_basis": "spectral_band_limited",
    }

    ranking_summary = _dynamic_sequence_summary_to_ranking_summary(
        dynamic_summary,
        static_summary,
    )

    assert ranking_summary["safe_for_ordering"] is True
    assert ranking_summary["safe_for_fusion"] is False
    assert ranking_summary["safe_for_dynamic_joint_fusion_source"] is True
    assert "joint dynamic estimator" in ranking_summary["status_reason"]


def test_sem_transport_step_resolves_surface_escape_before_vacuum_stopping() -> None:
    from imaging_models.sem_backends.physical_transport import (
        _sem_transport_step_event_lengths,
    )

    step_nm, exited, stopped = _sem_transport_step_event_lengths(
        sampled_step_nm=np.array([20.0]),
        distance_to_cutoff_nm=np.array([50.0]),
        z_nm=np.array([1.5]),
        uz=np.array([-1.0]),
    )

    assert np.isclose(float(step_nm[0]), 1.5)
    assert bool(exited[0])
    assert not bool(stopped[0])


def test_assignment_ambiguity_position_distance_is_projected_lateral_xy() -> None:
    from config import default_params
    from supervision_policy import compute_assignment_ambiguity_support

    params = default_params()
    params["supervision_ambiguity_distance_scale_nm"] = 100.0
    support, metadata = compute_assignment_ambiguity_support(
        0,
        np.array([0.0, 0.0, 0.0]),
        np.ones((3, 3), dtype=bool),
        all_positions_nm=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10_000.0],
            ]
        ),
        params=params,
    )

    assert metadata["assignment_position_distance_basis"] == "lateral_xy_nm"
    assert np.isclose(float(metadata["nearest_competitor_distance_nm"]), 0.0)
    assert np.isclose(float(support), 0.5)


def test_information_layer_sigma_xy_uses_l2_not_rms() -> None:
    from fisher.information_object import InformationObject, allocate, rank
    from lab_fisher_report.information_layer import _best_k_fusion
    from lab_fisher_report.partial_order import MicroscopeCandidate

    # Covariance diag(9, 16) gives sigma_x=3 nm, sigma_y=4 nm.  The
    # paper-facing lateral scalar is the L2 total 2D bound, 5 nm; RMS is
    # sqrt(12.5) and must never be emitted as sigma_xy_nm.
    info = InformationObject(
        fisher=np.diag([1.0 / 9.0, 1.0 / 16.0]),
        axes=("x", "y"),
        label="scope_a",
        modality="bright_field",
    )

    assert np.isclose(info.sigma_l2(("x", "y")), 5.0)
    assert np.isclose(info.sigma_rms(("x", "y")), np.sqrt(12.5))
    assert not hasattr(info, "sigma_joint")

    fusion_rows = _best_k_fusion([info], max_k=1, axes=("x", "y"))
    assert np.isclose(float(fusion_rows[0]["fusion_sigma_xy_nm"]), 5.0)

    candidate = MicroscopeCandidate(name="scope_a", info=info, modality="bright_field")
    assert np.isclose(candidate.sigma_xy_nm(), 5.0)

    ranking = rank([info], axes=("x", "y"))
    assert np.isclose(float(ranking[0]["sigma_l2_nm"]), 5.0)
    assert np.isclose(float(ranking[0]["sigma_rms_nm"]), np.sqrt(12.5))

    allocation = allocate([info], axes=("x", "y"), prune_dominated=False)
    assert np.isclose(float(allocation["sigma_l2_nm"]), 5.0)
    assert np.isclose(float(allocation["sigma_rms_nm"]), np.sqrt(12.5))


def test_full_vector_analytic_polarizability_preserves_transverse_dipole_component() -> None:
    from config import default_params
    from optical_scattering import optical_scattering_render_multiplier
    from particle_specs import ParticleComponentSpec

    params = default_params()
    params["optical_scattering_model"] = "analytic_polarizability"
    params["polarization_model"] = "linear_x"
    params["vectorial_polarization_rotation_deg"] = 0.0
    params["vectorial_detection_mode"] = "full_vector"
    refractive_index = 1.6 + 0.02j
    component = ParticleComponentSpec(
        shape="ellipsoid",
        offset_nm=(0.0, 0.0, 0.0),
        diameter_nm=120.0,
        axes_nm=(120.0, 30.0, 30.0),
        refractive_index=refractive_index,
    )
    angle = np.deg2rad(45.0)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotation_z = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    vector_metadata = {
        "field_representation": "vectorial_coherent_field",
        "scalar_compatibility_reduction": "full_vector_field",
        "optical_scattering_model": "analytic_polarizability",
        "particle_refractive_index": {
            "real": float(refractive_index.real),
            "imag": float(refractive_index.imag),
        },
    }

    full_vector = optical_scattering_render_multiplier(
        params,
        component_geometry=component,
        material_properties=None,
        orientation_matrix=rotation_z,
        field_metadata=vector_metadata,
    )

    assert isinstance(full_vector, np.ndarray)
    assert full_vector.shape == (3,)
    assert abs(complex(full_vector[1])) > 1.0e-12 * max(abs(complex(full_vector[0])), 1.0)
    assert float(np.sum(np.abs(full_vector) ** 2)) > abs(complex(full_vector[0])) ** 2

    analyzer_params = dict(params)
    analyzer_params["vectorial_detection_mode"] = "analyzer_x"
    analyzer_x = optical_scattering_render_multiplier(
        analyzer_params,
        component_geometry=component,
        material_properties=None,
        orientation_matrix=rotation_z,
        field_metadata=vector_metadata,
    )
    assert np.isclose(complex(analyzer_x), complex(full_vector[0]))

    scalar_metadata = {
        "field_representation": "scalar_coherent_vector_component",
        "scalar_compatibility_reduction": "analyzer_x_component",
        "optical_scattering_model": "analytic_polarizability",
        "particle_refractive_index": vector_metadata["particle_refractive_index"],
    }
    scalar_fallback = optical_scattering_render_multiplier(
        params,
        component_geometry=component,
        material_properties=None,
        orientation_matrix=rotation_z,
        field_metadata=scalar_metadata,
    )
    assert np.isclose(abs(complex(scalar_fallback)), float(np.linalg.norm(full_vector)))
