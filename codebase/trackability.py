from __future__ import annotations

import numpy as np

from trajectory import (
    stokes_einstein_diffusion_coefficient,
    resolve_translational_diameters_nm,
)

_TEMPORAL_SUPPORT_SIGMA_RADIUS = 3.0


class TrackabilityModel:
    """
    Computes temporal support for video supervision.

    This module is temporal-only. Detector-noise and signal-support
    calculations are handled by supervision_policy.py and camera_noise.py.
    """

    def __init__(self, params: dict, num_particles: int):
        self.params = params
        self.num_particles = int(num_particles)
        self.dt = 1.0 / float(params['fps'])

        translational_diameters_nm = resolve_translational_diameters_nm(params)
        if translational_diameters_nm.shape[0] != self.num_particles:
            raise ValueError(
                'Length of translational diameters array must match '
                f'num_particles ({self.num_particles}). Got '
                f'{translational_diameters_nm.shape[0]}.'
            )

        temp_K = float(params['temperature_K'])
        viscosity = float(params['viscosity_Pa_s'])

        self.diffusion_coefficients_m2_s = np.zeros(self.num_particles, dtype=float)
        self.r_sigma_nm = np.zeros(self.num_particles, dtype=float)

        for i in range(self.num_particles):
            D_m2_s = stokes_einstein_diffusion_coefficient(
                translational_diameters_nm[i], temp_K, viscosity
            )
            self.diffusion_coefficients_m2_s[i] = D_m2_s
            self.r_sigma_nm[i] = np.sqrt(4.0 * D_m2_s * self.dt) * 1e9

        self.last_positions_nm = [None] * self.num_particles
        self.lost = np.zeros(self.num_particles, dtype=bool)

    def reset(self) -> None:
        self.last_positions_nm = [None] * self.num_particles
        self.lost[:] = False

    def is_particle_lost(self, particle_index: int) -> bool:
        return bool(self.lost[int(particle_index)])

    def are_all_particles_lost(self) -> bool:
        return bool(np.all(self.lost))

    def mark_lost(self, particle_index: int) -> None:
        self.lost[int(particle_index)] = True

    def update_and_compute(
        self,
        particle_index: int,
        frame_index: int,
        position_nm: np.ndarray,
    ) -> float:
        """Update the stored position and return Brownian temporal support."""
        del frame_index
        particle_index = int(particle_index)
        if self.lost[particle_index]:
            return 0.0

        position_nm = np.asarray(position_nm, dtype=float)
        if position_nm.shape != (3,):
            raise ValueError(
                'position_nm must be a 1D array of shape (3,) representing [x, y, z] in nm.'
            )

        last_pos = self.last_positions_nm[particle_index]
        if last_pos is None:
            temporal_support = 1.0
        else:
            delta_xy_nm = position_nm[:2] - last_pos[:2]
            r_nm = float(np.linalg.norm(delta_xy_nm))
            sigma_r_nm = float(self.r_sigma_nm[particle_index])

            if sigma_r_nm <= 0.0:
                temporal_support = 1.0 if r_nm == 0.0 else 0.0
            else:
                # Three radial standard deviations give a permissive temporal
                # support envelope before assignment confidence decays.
                scaled_r = r_nm / (_TEMPORAL_SUPPORT_SIGMA_RADIUS * sigma_r_nm)
                temporal_support = float(np.exp(-0.5 * scaled_r**2))
                temporal_support = max(0.0, min(1.0, temporal_support))

        self.last_positions_nm[particle_index] = position_nm.copy()
        return temporal_support
