"""V22  Rotational + confined diffusion (extends v04's free translational MSD).

Black-box validation of codebase/trajectory.py via its PUBLIC API:

    simulate_orientations(params, num_particles, num_frames)
    simulate_trajectories(params)
    resolve_rotational_step_std_rad(params, num_particles)
    stokes_einstein_diffusion_coefficient(diameter_nm, temp_K, viscosity_Pa_s)

Reuses config.PARAMS + overrides and the deterministic seed pattern from v04.
v04 only checked FREE translational MSD slope == 2*dim*D.  This test adds the
two pieces v04 did not exercise: ROTATIONAL diffusion, and CONFINED (bounded)
translational diffusion.

--------------------------------------------------------------------------------
PART 1 -- ROTATIONAL diffusion (Stokes-Einstein-Debye)
--------------------------------------------------------------------------------
Enable rotational diffusion in stokes_einstein mode.  simulate_orientations
returns (P, T, 3, 3) SO(3) rotation matrices.  We form the relative rotation
between frames separated by lag L,

    dR = R(t)^T @ R(t+L),   theta = arccos( clip((trace(dR) - 1) / 2) ),

and compute the rotational MSD <theta^2>(L).  For isotropic 3-D rotational
Brownian motion the rotation vector has three independent components, each of
variance 2*D_rot*t, so for small angles

    <theta^2> = 3 * (2 D_rot t) = 6 D_rot t            (2*D_rot per rotational DOF),

i.e. the MSD-vs-time slope = 6 * D_rot.  The analytic rotational Stokes-Einstein-
Debye coefficient is

    D_rot = k_B T / (8 pi eta r^3)     [rad^2 / s].

We choose a LARGE particle so D_rot is small and the per-frame angular step
(sqrt(2 D_rot dt)) stays well inside the small-angle regime; otherwise theta
saturates at pi and the slope is meaningless.  ASSERT: measured slope matches
6*D_rot within tolerance, and resolve_rotational_step_std_rad == sqrt(2 D_rot dt).
Units: angles in rad, MSD in rad^2, slope in rad^2/s, D_rot in rad^2/s.

--------------------------------------------------------------------------------
PART 2 -- CONFINED translational diffusion (substrate exclusion)
--------------------------------------------------------------------------------
Enable a gold-holes substrate pattern with a SMALL hole so the particle is
trapped inside a single circular fluid well surrounded by solid (reflecting
walls).  We compute the lateral (x,y) translational MSD exactly as v04 does and
compare it to the free-diffusion extrapolation 2*dim*D*t.  Free diffusion grows
linearly forever; confinement bounds the MSD (it plateaus).  ASSERT: at large
lag the confined MSD is far BELOW the free extrapolation (here < 25%), AND a
free control run in the SAME harness tracks the free extrapolation (ratio ~1).
The contrast between the two runs is the confinement signature.
Units: positions in nm, MSD in nm^2, D in nm^2/s.

Run:  python V22_rotational_confined_diffusion.py     (pure numpy, no external deps)
Deterministic via PARAMS['random_seed'].  EXERCISES the simulator (light/fast).
"""
from __future__ import annotations

import copy

import numpy as np

from common import add_paths, banner, verdict

add_paths()

from config import BOLTZMANN_CONSTANT, default_param_value, default_params  # noqa: E402
from trajectory import (  # noqa: E402
    resolve_rotational_step_std_rad,
    simulate_orientations,
    simulate_trajectories,
    stokes_einstein_diffusion_coefficient,
)

banner("V22  Rotational (SED) + confined (bounded) diffusion")

# Shared fluid / timing.
T_K = 298.15
ETA = 1.0e-3          # Pa s
FPS = 1000.0
SEED = 20240517


def _clone_particle(template, name, diameter_nm):
    p = copy.deepcopy(template)
    p["name"] = name
    p["motion"]["hydrodynamic_diameter_nm"] = float(diameter_nm)
    p["motion"]["initial_position_nm"] = None
    if p.get("components"):
        p["components"][0]["diameter_nm"] = float(diameter_nm)
    return p


def _make_particles(n, diameter_nm):
    template = default_param_value("particles")[0]
    return [_clone_particle(template, f"p{i}", diameter_nm) for i in range(n)]


def _rotational_msd(orientations, max_lag):
    """<theta^2>(lag) from a (P, T, 3, 3) orientation stack."""
    T = orientations.shape[1]
    max_lag = int(min(max_lag, T // 4))
    lags = np.arange(1, max_lag)
    msd = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        r0 = orientations[:, :-lag]            # (P, T-lag, 3, 3)
        r1 = orientations[:, lag:]
        # dR = R0^T R1  (relative rotation), batched.
        dR = np.einsum("ptij,ptik->ptjk", r0, r1)
        trace = np.trace(dR, axis1=2, axis2=3)
        cos_ang = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        msd[i] = float(np.mean(ang ** 2))
    return lags, msd


# ===========================================================================
# PART 1 -- ROTATIONAL diffusion
# ===========================================================================
ROT_DIAMETER_NM = 2000.0   # large bead -> small D_rot -> small-angle regime
ROT_NUM_FRAMES = 6000
ROT_NUM_PARTICLES = 4

p_rot = default_params()
p_rot.update(
    {
        "temperature_K": T_K,
        "viscosity_Pa_s": ETA,
        "fps": FPS,
        "num_frames": ROT_NUM_FRAMES,
        "random_seed": SEED,
        "rotational_diffusion_enabled": True,
        "rotational_diffusion_mode": "stokes_einstein",
        "sample_environment_enabled": False,
        "sample_environment_pattern_enabled": False,
    }
)
p_rot["particles"] = _make_particles(ROT_NUM_PARTICLES, ROT_DIAMETER_NM)

dt = 1.0 / FPS
r_m = ROT_DIAMETER_NM * 1e-9 / 2.0
D_rot_analytic = BOLTZMANN_CONSTANT * T_K / (8.0 * np.pi * ETA * r_m ** 3)  # rad^2/s
expected_step_std = np.sqrt(2.0 * D_rot_analytic * dt)                       # rad

# (1a) the resolver must reproduce sqrt(2 D_rot dt) per axis.
resolved_step_std = resolve_rotational_step_std_rad(p_rot, ROT_NUM_PARTICLES)
step_std_rel_err = float(
    np.max(np.abs(resolved_step_std - expected_step_std) / expected_step_std)
)
check_step = step_std_rel_err < 1.0e-9

# (1b) measured rotational MSD slope == 6 * D_rot.
ROT_MSD_REL_TOL = 0.10
orientations = simulate_orientations(p_rot, ROT_NUM_PARTICLES, ROT_NUM_FRAMES)
assert orientations is not None, "rotational diffusion returned None despite being enabled"
assert orientations.shape == (ROT_NUM_PARTICLES, ROT_NUM_FRAMES, 3, 3)
# sanity: frame 0 is identity, matrices are proper rotations.
assert np.allclose(orientations[:, 0], np.eye(3))
det0 = np.linalg.det(orientations[:, 5])
assert np.allclose(det0, 1.0, atol=1e-6), f"orientations not in SO(3): det={det0}"

# Restrict the fit to small lags so theta stays << pi (linear MSD regime).
lags_rot, msd_rot = _rotational_msd(orientations, max_lag=40)
t_rot = lags_rot * dt
slope_rot = np.polyfit(t_rot, msd_rot, 1)[0]          # rad^2/s
expected_slope_rot = 6.0 * D_rot_analytic             # 2 D_rot * 3 DOF
rot_rel_err = abs(slope_rot - expected_slope_rot) / expected_slope_rot
# Confirm we really were in the small-angle regime at the largest lag fitted.
max_theta2 = float(msd_rot[-1])
small_angle_ok = max_theta2 < (np.pi / 2.0) ** 2
check_rot = (rot_rel_err < ROT_MSD_REL_TOL) and small_angle_ok and check_step


# ===========================================================================
# PART 2 -- CONFINED translational diffusion
# ===========================================================================
CONF_DIAMETER_NM = 100.0
CONF_NUM_FRAMES = 3000
CONF_NUM_PARTICLES = 8
HOLE_DIAMETER_UM = 1.0     # 1 um hole -> ~500 nm confinement radius

D_trans = stokes_einstein_diffusion_coefficient(CONF_DIAMETER_NM, T_K, ETA)  # m^2/s
D_trans_nm2_s = D_trans * 1e18


def _lateral_msd(traj, max_lag):
    """Lateral (x,y) MSD averaged over particles and start times."""
    lat = traj[:, :, :2]                      # (P, T, 2) nm
    T = lat.shape[1]
    max_lag = int(min(max_lag, T // 4))
    lags = np.arange(1, max_lag)
    msd = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        disp = lat[:, lag:] - lat[:, :-lag]   # (P, T-lag, 2)
        msd[i] = float(np.mean(np.sum(disp ** 2, axis=-1)))
    return lags, msd


def _run_translational(confined):
    p = default_params()
    p.update(
        {
            "temperature_K": T_K,
            "viscosity_Pa_s": ETA,
            "fps": FPS,
            "num_frames": CONF_NUM_FRAMES,
            "random_seed": SEED,
            "rotational_diffusion_enabled": False,
        }
    )
    if confined:
        dims = copy.deepcopy(p["sample_environment_pattern_dimensions"])
        dims["hole_diameter_um"] = HOLE_DIAMETER_UM
        dims["hole_edge_to_edge_spacing_um"] = 2.0
        p.update(
            {
                "sample_environment_enabled": True,
                "sample_environment_pattern_enabled": True,
                "sample_environment_pattern": "gold_holes",
                "sample_environment_pattern_preset": "default_gold_holes",
                "sample_environment_exclusion_method": "reflection",
                # Perfect circular holes for a clean, deterministic geometry.
                "sample_environment_pattern_randomization_enabled": False,
                "sample_environment_pattern_shape_regularity": 1.0,
                "sample_environment_pattern_dimensions": dims,
            }
        )
    else:
        p.update(
            {
                "sample_environment_enabled": False,
                "sample_environment_pattern_enabled": False,
            }
        )
    p["particles"] = _make_particles(CONF_NUM_PARTICLES, CONF_DIAMETER_NM)
    return simulate_trajectories(p)


# Free control run (same harness) -- should track the free extrapolation.
traj_free = _run_translational(confined=False)
lags_f, msd_free = _lateral_msd(traj_free, max_lag=300)
t_f = lags_f * dt
free_extrap_f = 2.0 * 2 * D_trans_nm2_s * t_f
free_ratio_last = float(msd_free[-1] / free_extrap_f[-1])
# Free MSD must NOT be bounded: it should reach a large fraction of 2*dim*D*t.
check_free_unbounded = free_ratio_last > 0.6

# Confined run -- should be bounded well below the free extrapolation.
traj_conf = _run_translational(confined=True)
lags_c, msd_conf = _lateral_msd(traj_conf, max_lag=300)
t_c = lags_c * dt
free_extrap_c = 2.0 * 2 * D_trans_nm2_s * t_c
conf_ratio_last = float(msd_conf[-1] / free_extrap_c[-1])
conf_plateau = float(np.mean(msd_conf[-50:]))
# Confinement: large-lag MSD far below the free extrapolation (< 25%).
check_confined_bounded = conf_ratio_last < 0.25
# Confined particle span should be comparable to the hole, not the FOV.
conf_span_nm = float(
    np.max(np.ptp(traj_conf[:, :, :2], axis=1))
)
check_span = conf_span_nm < (HOLE_DIAMETER_UM * 1000.0 * 1.2)

check_conf = check_free_unbounded and check_confined_bounded and check_span


# ===========================================================================
# Report.
# ===========================================================================
print("PART 1 -- ROTATIONAL diffusion (Stokes-Einstein-Debye)")
print(f"  particle diameter          : {ROT_DIAMETER_NM:.0f} nm   "
      f"particles: {ROT_NUM_PARTICLES}   frames: {ROT_NUM_FRAMES}")
print(f"  D_rot analytic             : {D_rot_analytic:.6e} rad^2/s")
print(f"  per-axis step std expected : {expected_step_std:.6e} rad  (sqrt(2 D_rot dt))")
print(f"  per-axis step std resolved : {resolved_step_std[0]:.6e} rad   rel.err {step_std_rel_err:.2e}")
print(f"  MSD slope measured         : {slope_rot:.6e} rad^2/s")
print(f"  6*D_rot expected           : {expected_slope_rot:.6e} rad^2/s")
print(f"  rotational rel. error      : {rot_rel_err:.4f}   (max theta^2 fitted = {max_theta2:.4f} rad^2)")
print(f"  step-std resolver match    : {'PASS' if check_step else 'FAIL'}")
print(f"  small-angle regime ok      : {'PASS' if small_angle_ok else 'FAIL'}")
print(f"  PART 1 verdict             : {'PASS' if check_rot else 'FAIL'}  (slope within {100.0 * ROT_MSD_REL_TOL:.0f}% of 6*D_rot)")
print()
print("PART 2 -- CONFINED translational diffusion (gold-holes exclusion)")
print(f"  particle diameter          : {CONF_DIAMETER_NM:.0f} nm   "
      f"particles: {CONF_NUM_PARTICLES}   frames: {CONF_NUM_FRAMES}")
print(f"  hole diameter              : {HOLE_DIAMETER_UM:.2f} um")
print(f"  D_trans                    : {D_trans_nm2_s:.4e} nm^2/s")
print(f"  FREE   MSD/2dimDt @ max lag: {free_ratio_last:.4f}   (unbounded, want > 0.6)")
print(f"  CONFINED MSD/2dimDt @ max  : {conf_ratio_last:.4f}   (bounded,   want < 0.25)")
print(f"  CONFINED MSD plateau       : {conf_plateau:.4e} nm^2")
print(f"  confined lateral span      : {conf_span_nm:.1f} nm  (< {HOLE_DIAMETER_UM*1000:.0f} nm hole => trapped)")
print(f"  free unbounded             : {'PASS' if check_free_unbounded else 'FAIL'}")
print(f"  confined bounded           : {'PASS' if check_confined_bounded else 'FAIL'}")
print(f"  trapped within hole        : {'PASS' if check_span else 'FAIL'}")
print(f"  PART 2 verdict             : {'PASS' if check_conf else 'FAIL'}")
print()
print("-" * 78)
print(f"{'check':<44}{'result':>10}")
print("-" * 78)
print(f"{'rotational MSD slope == 6*D_rot':<44}{'PASS' if check_rot else 'FAIL':>10}")
print(f"{'confinement bounds translational MSD':<44}{'PASS' if check_conf else 'FAIL':>10}")
print("-" * 78)

ok = check_rot and check_conf
raise SystemExit(
    verdict(
        ok,
        "(rotational MSD slope = 6*D_rot = 2*D_rot/DOF; "
        "substrate exclusion bounds translational MSD vs free 2*dim*D*t)",
    )
)
