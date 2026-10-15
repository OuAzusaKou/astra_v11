"""Bounded random perturbations for real-robot EE action baselines."""

from __future__ import annotations

from dataclasses import dataclass
import secrets

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EEPerturbation:
    left_translation_m: np.ndarray
    right_translation_m: np.ndarray
    left_rotation_rotvec: np.ndarray
    right_rotation_rotvec: np.ndarray


class RandomEEActionPerturber:
    """Apply one smoothly ramped rigid offset per arm and action chunk.

    The first waypoint receives ``1 / T`` of the sampled offset and the last
    receives the full offset. A fresh pair of offsets is sampled for the next
    policy chunk.
    """

    def __init__(
        self,
        position_cm: float,
        rotation_deg: float,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(position_cm) or position_cm < 0:
            raise ValueError("position_cm must be finite and non-negative")
        if not np.isfinite(rotation_deg) or rotation_deg < 0:
            raise ValueError("rotation_deg must be finite and non-negative")
        if position_cm == 0 and rotation_deg == 0:
            raise ValueError("at least one perturbation magnitude must be positive")

        self.position_m = float(position_cm) / 100.0
        self.rotation_rad = np.deg2rad(float(rotation_deg))
        self.seed = int(seed) if seed is not None else secrets.randbits(63)
        self._rng = np.random.default_rng(self.seed)

    def _fixed_norm_vector(self, magnitude: float) -> np.ndarray:
        if magnitude == 0:
            return np.zeros(3, dtype=np.float64)
        while True:
            direction = self._rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm > np.finfo(np.float64).eps:
                return direction * (magnitude / norm)

    def apply(self, arm_action: np.ndarray) -> tuple[np.ndarray, EEPerturbation]:
        action = np.asarray(arm_action)
        if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != 12:
            raise ValueError(f"arm_action must have shape (T, 12), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("arm_action contains NaN or infinity")

        output = np.asarray(action, dtype=np.float64).copy()
        ramp = (
            (np.arange(output.shape[0], dtype=np.float64) + 1.0)
            / float(output.shape[0])
        )[:, None]
        translations = [
            self._fixed_norm_vector(self.position_m),
            self._fixed_norm_vector(self.position_m),
        ]
        rotation_rotvecs = [
            self._fixed_norm_vector(self.rotation_rad),
            self._fixed_norm_vector(self.rotation_rad),
        ]

        for xyz_start, rot_start, translation, rotation_rotvec in (
            (0, 3, translations[0], rotation_rotvecs[0]),
            (6, 9, translations[1], rotation_rotvecs[1]),
        ):
            output[:, xyz_start : xyz_start + 3] += ramp * translation
            action_rotation = Rotation.from_rotvec(
                output[:, rot_start : rot_start + 3]
            )
            noise_rotation = Rotation.from_rotvec(ramp * rotation_rotvec)
            output[:, rot_start : rot_start + 3] = (
                noise_rotation * action_rotation
            ).as_rotvec()

        return output.astype(action.dtype, copy=False), EEPerturbation(
            left_translation_m=translations[0],
            right_translation_m=translations[1],
            left_rotation_rotvec=rotation_rotvecs[0],
            right_rotation_rotvec=rotation_rotvecs[1],
        )


def format_ee_perturbation(perturbation: EEPerturbation) -> str:
    left_cm = perturbation.left_translation_m * 100.0
    right_cm = perturbation.right_translation_m * 100.0
    left_deg = np.rad2deg(perturbation.left_rotation_rotvec)
    right_deg = np.rad2deg(perturbation.right_rotation_rotvec)
    return (
        f"left_xyz_cm={np.round(left_cm, 3).tolist()} "
        f"left_rotvec_deg={np.round(left_deg, 3).tolist()} "
        f"right_xyz_cm={np.round(right_cm, 3).tolist()} "
        f"right_rotvec_deg={np.round(right_deg, 3).tolist()}"
    )
