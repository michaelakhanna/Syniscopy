import sys, numpy as np
sys.path.insert(0,"codebase")
from copy import deepcopy
from config import PARAMS, BOLTZMANN_CONSTANT
from trajectory import simulate_trajectories, stokes_einstein_diffusion_coefficient

# ---- Clean free-diffusion config (unconstrained z, NO substrate) ----
d_nm, T, eta, fps = 100.0, 298.0, 1.0e-3, 1000.0
P, F, seed = 64, 4000, 12345
dt = 1.0/fps

p = deepcopy(PARAMS)
p.update({
    "random_seed": seed, "fps": fps, "num_frames": F,
    "temperature_K": T, "viscosity_Pa_s": eta,
    "z_motion_constraint_model": "unconstrained",
    "sample_environment_enabled": False,
    "sample_environment_pattern_enabled": False,
    "sample_environment_pattern": "none",
    "sample_environment_pattern_preset": "empty_background",
    "image_size_pixels": 256, "pixel_size_nm": 100.0,
    "initial_z_span_nm": 4000.0,
})
# one identical particle template per slot
one = {"name":"p","motion":{"hydrodynamic_diameter_nm":d_nm,"initial_position_nm":None},
       "signal_multiplier":1.0,"source_multiplier":1.0,
       "components":[{"shape":"sphere","offset_nm":[0,0,0],"diameter_nm":d_nm,
                      "material":"polystyrene","refractive_index":None,
                      "signal_multiplier":1.0,"source_multiplier":1.0,"material_properties":None}]}
p["particles"] = [deepcopy(one) for _ in range(P)]

D = stokes_einstein_diffusion_coefficient(d_nm, T, eta)        # m^2/s
sigma_step_nm = np.sqrt(2.0*D*dt)*1e9                          # predicted per-axis step std
print(f"D = {D:.6e} m^2/s ; predicted per-step per-axis std = {sigma_step_nm:.4f} nm")

traj = simulate_trajectories(p)                                # (P,F,3) nm
print("trajectory shape:", traj.shape)

# reproducibility under same seed
traj2 = simulate_trajectories(p)
print("seed reproducible (identical re-run):", bool(np.array_equal(traj, traj2)))

# ---- Check 1: per-step displacement std vs sqrt(2 D dt) ----
steps = np.diff(traj, axis=1)                                  # (P,F-1,3) nm
emp_std = steps.reshape(-1,3).std(axis=0)
print("\n[Check1] per-axis step std (nm):  x=%.3f y=%.3f z=%.3f  vs predicted %.3f"
      %(emp_std[0],emp_std[1],emp_std[2],sigma_step_nm))
print("         rel error: x=%.3f%% y=%.3f%% z=%.3f%%"
      %(*(100*abs(emp_std/sigma_step_nm-1)),))

# ---- Check 2: D recovered from step variance, per axis: var=2 D dt ----
D_emp = (emp_std*1e-9)**2/(2*dt)                               # m^2/s
print("\n[Check2] D recovered per axis (m^2/s): x=%.4e y=%.4e z=%.4e  vs %.4e"
      %(D_emp[0],D_emp[1],D_emp[2],D))
print("         rel error: x=%.3f%% y=%.3f%% z=%.3f%%"
      %(*(100*abs(D_emp/D-1)),))

# ---- Check 3: ensemble MSD vs lag, slope should be 2 D per axis (6 D in 3D) ----
# time-and-ensemble averaged MSD over a set of lags
lags = np.unique(np.round(np.geomspace(1, F-1, 25)).astype(int))
msd = []
for L in lags:
    disp = traj[:, L:, :] - traj[:, :-L, :]                   # (P, F-L, 3) nm
    msd.append((disp**2).sum(axis=2).mean())                  # 3D MSD, nm^2
msd = np.array(msd)
tau = lags*dt
# fit MSD = slope*tau  (force through ~0); slope should be 6 D (3D), units nm^2/s
slope = np.polyfit(tau, msd, 1)[0]                            # nm^2/s
D_msd = (slope*1e-18)/6.0                                     # m^2/s
print("\n[Check3] 3D MSD slope -> D = %.4e m^2/s  vs %.4e  (rel %.2f%%)"
      %(D_msd, D, 100*abs(D_msd/D-1)))

# ---- Verdict ----
ok1 = np.all(np.abs(emp_std/sigma_step_nm-1) < 0.05)
ok3 = abs(D_msd/D-1) < 0.10
print("\nVERDICT: step-std<5%%: %s ; MSD-slope D<10%%: %s ; seed-reproducible: %s"
      %(ok1, ok3, bool(np.array_equal(traj,traj2))))
