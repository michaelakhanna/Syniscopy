"""V21  Dynamic-Bayesian sequence CRLB vs an independent analytic Kalman filter.

Black-box validation of codebase/fisher/dynamic_bayesian.py via its PUBLIC API:

    build_brownian_process_covariance(state_axes, ...)
    sequence_sum_fisher_to_crlb(per_frame_fisher)
    compute_dynamic_bayesian_crlb_from_fisher_sequence(per_frame_fisher_matrices,
                                                       process_noise_covariance, ...)
    summarize_fisher_sequence(...)

WHAT IS COMPARED
----------------
Syniscopy treats a frame sequence as a linear-Gaussian estimation problem and
runs a Bayesian information filter to produce a per-frame "dynamic CRLB"
(posterior parameter variance). We re-derive the SAME quantity independently,
from scratch, with a textbook Kalman information filter (state transition A = I):

    P_pred = A P_post A^T + Q          # prediction inflates covariance by Q
    J_pred = inv(P_pred)               # prior information
    J_post = J_pred + H_t              # information-form measurement update
    P_post = inv(J_post)               # posterior covariance
    dynamic CRLB_t = diag(P_post)      # per-axis variance lower bound

We feed both implementations identical per-frame Fisher matrices H_t and an
identical Brownian process-noise covariance Q, then compare element by element.

FOUR INDEPENDENT CHECKS (all must pass)
---------------------------------------
  (A) Syniscopy's dynamic CRLB sequence == our independent Kalman covariance
      diagonal, to ~1e-9 (we observe ~1e-15).
  (B) Static sum-of-Fisher CRLB (sequence_sum_fisher_to_crlb, no process noise)
      == the zero-process-noise limit of the dynamic filter.  Diffusion off =>
      the recursive filter collapses to the cumulative-Fisher estimator.
  (C) The dynamic CRLB monotonically TIGHTENS as frames accumulate (information
      only grows when Q = 0).
  (D) With NONZERO process noise the dynamic CRLB is strictly LOOSER than the
      static sum and reaches a finite steady state: diffusion erodes information
      between frames, so you cannot keep shrinking the bound forever.  This is
      the meaningful physical check that distinguishes a genuine dynamic filter
      from naive Fisher accumulation.

Units: per-frame Fisher H is in (1/state_unit^2); state axes here are ("x","y")
in nm, so CRLB / covariance diagonals are in nm^2.  Q (build_brownian_process_
covariance) returns nm^2 per frame for translational diffusion (2 D dt, m->nm).

Run:  python V21_dynamic_bayesian_kalman.py     (pure numpy, no external deps)
Deterministic: a fixed RNG seed builds the synthetic Fisher sequence.
This test does NOT touch optics/rendering; it exercises only the estimator layer.
"""
from __future__ import annotations

import numpy as np

from common import add_paths, banner, verdict

add_paths()

from fisher.dynamic_bayesian import (  # noqa: E402  (after add_paths)
    build_brownian_process_covariance,
    compute_dynamic_bayesian_crlb_from_fisher_sequence,
    sequence_sum_fisher_to_crlb,
    summarize_fisher_sequence,
)

banner("V21  Dynamic-Bayesian sequence CRLB vs independent Kalman information filter")

# ---------------------------------------------------------------------------
# 0.  Build a deterministic synthetic per-frame Fisher sequence (2x2 PSD).
#     Two regimes: a constant-Fisher sequence (clean monotonicity / steady
#     state) and a varying-Fisher sequence (stresses the general recursion).
# ---------------------------------------------------------------------------
STATE_AXES = ("x", "y")
FPS = 50.0
N_FRAMES = 40
SEED = 20240517

rng = np.random.default_rng(SEED)


def _random_psd_2x2(generator: np.random.Generator) -> np.ndarray:
    """A random symmetric positive-definite 2x2 information matrix."""
    a = generator.normal(size=(2, 2))
    return a @ a.T + 0.75 * np.eye(2)


# Constant per-frame information (e.g. identical exposures of a static emitter).
H_const = np.array([[4.0, 0.5], [0.5, 3.0]], dtype=float)
seq_const = [H_const.copy() for _ in range(N_FRAMES)]

# Varying per-frame information (e.g. fluctuating signal / focus per frame).
seq_vary = [_random_psd_2x2(rng) for _ in range(N_FRAMES)]

# A very large (but finite) initial covariance => an essentially uninformative
# prior, so the dynamic filter's Q=0 limit collapses onto the static sum.
P0 = np.eye(2) * 1.0e12

# Brownian process-noise covariances.
Q_zero = build_brownian_process_covariance(
    STATE_AXES, fps=FPS, translational_diffusion_coeff_m2_s=0.0
)
D_TRANS_M2_S = 5.0e-13  # nonzero translational diffusion -> Q > 0
Q_nonzero = build_brownian_process_covariance(
    STATE_AXES, fps=FPS, translational_diffusion_coeff_m2_s=D_TRANS_M2_S
)


# ---------------------------------------------------------------------------
# 1.  Independent reference: textbook Kalman information filter (A = I).
#     ~20 lines, no Syniscopy code involved.
# ---------------------------------------------------------------------------
def reference_kalman_crlb(
    per_frame_fisher, process_noise_cov, initial_cov
):
    """Posterior covariance diagonal per frame from a Kalman information filter.

    P_pred = A P A^T + Q ; J_post = inv(P_pred) + H ; P = inv(J_post).
    A is the identity (random-walk state model matching Q's construction).
    Returns an (N, dim) array of per-axis posterior variances (the dynamic CRLB).
    """
    q = np.asarray(process_noise_cov, dtype=float)
    p = np.asarray(initial_cov, dtype=float).copy()
    diag_seq = []
    for h in per_frame_fisher:
        p_pred = p + q                       # predict (A = I)
        j_post = np.linalg.inv(p_pred) + np.asarray(h, dtype=float)
        p = np.linalg.inv(j_post)            # posterior covariance
        diag_seq.append(np.diag(p).copy())
    return np.asarray(diag_seq, dtype=float)


def syn_dynamic_crlb(seq, q):
    """Pull the per-frame dynamic CRLB sequence out of the Syniscopy API."""
    out = compute_dynamic_bayesian_crlb_from_fisher_sequence(
        seq,
        q,
        state_axes=STATE_AXES,
        initial_covariance=P0,
        include_fisher_matrices=True,
    )
    return np.asarray(out["dynamic_crlb"], dtype=float), out


# ---------------------------------------------------------------------------
# CHECK (A) -- Syniscopy dynamic CRLB == independent Kalman, both regimes,
#              both Q = 0 and Q > 0.
# ---------------------------------------------------------------------------
TOL_KALMAN = 1.0e-9
kalman_errors = {}
for label, seq in (("const", seq_const), ("vary", seq_vary)):
    for qlabel, q in (("Q=0", Q_zero), ("Q>0", Q_nonzero)):
        syn, _ = syn_dynamic_crlb(seq, q)
        ref = reference_kalman_crlb(seq, q, P0)
        err = float(np.max(np.abs(syn - ref)))
        kalman_errors[f"{label}/{qlabel}"] = err
check_A = all(e < TOL_KALMAN for e in kalman_errors.values())

# ---------------------------------------------------------------------------
# CHECK (B) -- static sum-of-Fisher CRLB == dynamic CRLB in the Q=0 limit.
# ---------------------------------------------------------------------------
_, static_cov, _ = sequence_sum_fisher_to_crlb(seq_vary)
static_crlb_seq = np.asarray([np.diag(c) for c in static_cov], dtype=float)
syn_dyn_zeroQ, _ = syn_dynamic_crlb(seq_vary, Q_zero)
# Compare on the well-conditioned later frames (the very first frames carry the
# uninformative-prior transient; by frame ~3 the prior is negligible).
static_vs_dynamic_err = float(
    np.max(np.abs(static_crlb_seq[3:] - syn_dyn_zeroQ[3:]))
)
TOL_STATIC = 1.0e-6
check_B = static_vs_dynamic_err < TOL_STATIC

# ---------------------------------------------------------------------------
# CHECK (C) -- dynamic CRLB monotonically tightens (Q = 0, constant Fisher).
# ---------------------------------------------------------------------------
syn_dyn_const_zeroQ, _ = syn_dynamic_crlb(seq_const, Q_zero)
# Each axis variance must be non-increasing frame to frame (allow tiny slack).
diffs = np.diff(syn_dyn_const_zeroQ, axis=0)
check_C = bool(np.all(diffs <= 1.0e-12))
final_over_initial = syn_dyn_const_zeroQ[-1] / syn_dyn_const_zeroQ[0]

# ---------------------------------------------------------------------------
# CHECK (D) -- nonzero process noise => dynamic CRLB strictly LOOSER than the
#              static sum, and reaches a finite steady state.
# ---------------------------------------------------------------------------
_, static_cov_c, _ = sequence_sum_fisher_to_crlb(seq_const)
static_crlb_const_final = np.diag(static_cov_c[-1])
syn_dyn_const_nonzeroQ, summary_out = syn_dynamic_crlb(seq_const, Q_nonzero)
dyn_final_nonzeroQ = syn_dyn_const_nonzeroQ[-1]
# (i) looser than the static sum (per axis), with a real margin.
looser = bool(np.all(dyn_final_nonzeroQ > static_crlb_const_final * 1.5))
# (ii) reached a steady state (last two frames essentially equal).
steady_state_change = float(
    np.max(np.abs(syn_dyn_const_nonzeroQ[-1] - syn_dyn_const_nonzeroQ[-2]))
)
steady = steady_state_change < 1.0e-6
check_D = looser and steady

# ---------------------------------------------------------------------------
# Cross-check the higher-level summarize_fisher_sequence() public wrapper too:
# its static_crlb_final must equal the standalone sequence_sum_fisher_to_crlb
# and it must surface a dynamic_bayesian block when enabled.
# ---------------------------------------------------------------------------
summary = summarize_fisher_sequence(
    seq_const,
    state_axes=STATE_AXES,
    dynamic_bayesian_enabled=True,
    dynamic_process_noise_covariance=Q_nonzero,
    fps=FPS,
    initial_covariance=P0,
)
summary_static_final = np.asarray(summary["static_crlb_final"], dtype=float)
summary_match = float(np.max(np.abs(summary_static_final - static_crlb_const_final)))
check_E = (summary_match < 1.0e-6) and ("dynamic_bayesian_crlb" in summary)

# ---------------------------------------------------------------------------
# Report table.
# ---------------------------------------------------------------------------
print(f"state axes              : {STATE_AXES}   units: nm  (CRLB in nm^2)")
print(f"frames per sequence     : {N_FRAMES}    fps: {FPS}    seed: {SEED}")
print(f"translational D (Q>0)   : {D_TRANS_M2_S:.3e} m^2/s")
print(f"Q (per frame, nonzero)  : diag = {np.diag(Q_nonzero)}  nm^2/frame")
print()
print("-" * 78)
print(f"{'CHECK':<46}{'metric':>20}{'':>4}")
print("-" * 78)
for k, e in kalman_errors.items():
    print(f"(A) dyn CRLB vs indep Kalman  [{k:<9}] {'max|Δ|=':>16}{e:>11.2e}")
print(f"(B) static sum == dynamic(Q=0)           {'max|Δ|=':>16}{static_vs_dynamic_err:>11.2e}")
print(f"(C) dynamic CRLB monotonically tightens  {'max Δ(t)=':>16}{float(np.max(diffs)):>11.2e}")
print(f"    final/initial variance ratio (x,y)   {'':>16}  {np.round(final_over_initial, 5)}")
print(f"(D) nonzero-Q dynamic LOOSER than static {'':>16}")
print(f"    static sum final (nm^2)              {'':>16}  {np.round(static_crlb_const_final, 6)}")
print(f"    dynamic Q>0 final (nm^2)             {'':>16}  {np.round(dyn_final_nonzeroQ, 6)}")
print(f"    ratio dynamic/static (>1.5 req.)     {'':>16}  {np.round(dyn_final_nonzeroQ / static_crlb_const_final, 3)}")
print(f"    steady-state |ΔCRLB| last step       {'':>16}  {steady_state_change:.2e}")
print(f"(E) summarize_fisher_sequence consistency{'max|Δ|=':>16}{summary_match:>11.2e}")
print("-" * 78)
print()
print(f"(A) independent Kalman match     : {'PASS' if check_A else 'FAIL'}  (tol {TOL_KALMAN:g})")
print(f"(B) static == dynamic(Q=0) limit : {'PASS' if check_B else 'FAIL'}  (tol {TOL_STATIC:g})")
print(f"(C) monotonic tightening (Q=0)   : {'PASS' if check_C else 'FAIL'}")
print(f"(D) Q>0 looser + steady state    : {'PASS' if check_D else 'FAIL'}")
print(f"(E) public wrapper consistency   : {'PASS' if check_E else 'FAIL'}")

ok = check_A and check_B and check_C and check_D and check_E
raise SystemExit(
    verdict(
        ok,
        "(dynamic-Bayesian CRLB matches independent Kalman; "
        "Q=0 limit == static sum; tightening; Q>0 erodes information)",
    )
)
