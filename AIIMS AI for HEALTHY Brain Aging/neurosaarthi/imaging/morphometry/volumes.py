"""Native-geometry regional volumes and longitudinal change."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neurosaarthi.core.errors import DataValidationError


@dataclass(frozen=True)
class AnatomyLabels:
    left_hippocampus: tuple[int, ...] = (1,)
    right_hippocampus: tuple[int, ...] = (2,)
    ventricles: tuple[int, ...] = (3,)
    brain_tissue: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = [item for group in asdict(self).values() for item in group]
        if any(int(item) <= 0 for item in values):
            raise DataValidationError("Anatomical labels must be positive integers; zero is background")
        if len(values) != len(set(values)):
            raise DataValidationError("Anatomical label groups must not overlap")


@dataclass(frozen=True)
class MorphometryResult:
    voxel_volume_mm3: float
    left_hippocampal_volume_mm3: float
    right_hippocampal_volume_mm3: float
    total_hippocampal_volume_mm3: float
    hippocampal_asymmetry_index: float
    ventricular_volume_mm3: float
    brain_tissue_volume_mm3: float | None
    intracranial_volume_mm3: float | None
    normalized_hippocampal_volume: float | None
    normalized_ventricular_volume: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LongitudinalVolumeChange:
    baseline_volume_mm3: float
    followup_volume_mm3: float
    interval_days: float
    absolute_change_mm3: float
    percentage_change: float
    annualized_atrophy_percent: float


def compute_morphometry(
    segmentation: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    *,
    labels: AnatomyLabels | None = None,
    intracranial_mask: np.ndarray | None = None,
) -> MorphometryResult:
    """Calculate regional volumes from a discrete 3D segmentation."""

    mask = np.asarray(segmentation)
    if mask.ndim != 3 or any(size <= 0 for size in mask.shape):
        raise DataValidationError("Segmentation must be a non-empty 3D array")
    if not np.isfinite(mask).all() or not np.allclose(mask, np.rint(mask)):
        raise DataValidationError("Segmentation labels must be finite integers")
    if any(not np.isfinite(item) or item <= 0 for item in voxel_spacing_mm):
        raise DataValidationError("voxel_spacing_mm must contain three finite positive values")
    active = labels or AnatomyLabels()
    voxel_volume = float(np.prod(voxel_spacing_mm))
    left = _label_volume(mask, active.left_hippocampus, voxel_volume)
    right = _label_volume(mask, active.right_hippocampus, voxel_volume)
    total = left + right
    asymmetry = float((right - left) / total) if total > 0 else float("nan")
    ventricles = _label_volume(mask, active.ventricles, voxel_volume)
    brain = _label_volume(mask, active.brain_tissue, voxel_volume) if active.brain_tissue else None

    icv = None
    normalized_hippocampus = None
    normalized_ventricles = None
    if intracranial_mask is not None:
        intracranial = np.asarray(intracranial_mask)
        if intracranial.shape != mask.shape:
            raise DataValidationError("Intracranial mask shape must match segmentation")
        if not np.isfinite(intracranial).all():
            raise DataValidationError("Intracranial mask contains non-finite values")
        icv = float(np.count_nonzero(intracranial > 0) * voxel_volume)
        if icv <= 0:
            raise DataValidationError("Intracranial mask is empty")
        normalized_hippocampus = total / icv
        normalized_ventricles = ventricles / icv

    return MorphometryResult(
        voxel_volume_mm3=voxel_volume,
        left_hippocampal_volume_mm3=left,
        right_hippocampal_volume_mm3=right,
        total_hippocampal_volume_mm3=total,
        hippocampal_asymmetry_index=asymmetry,
        ventricular_volume_mm3=ventricles,
        brain_tissue_volume_mm3=brain,
        intracranial_volume_mm3=icv,
        normalized_hippocampal_volume=normalized_hippocampus,
        normalized_ventricular_volume=normalized_ventricles,
    )


def longitudinal_volume_change(
    baseline_volume_mm3: float, followup_volume_mm3: float, interval_days: float
) -> LongitudinalVolumeChange:
    """Return signed change and annualised atrophy (positive means loss)."""

    if not all(np.isfinite(value) for value in (baseline_volume_mm3, followup_volume_mm3, interval_days)):
        raise DataValidationError("Longitudinal volumes and interval must be finite")
    if baseline_volume_mm3 <= 0 or followup_volume_mm3 < 0:
        raise DataValidationError("Baseline volume must be positive and follow-up volume non-negative")
    if interval_days <= 0:
        raise DataValidationError("Longitudinal interval must be positive")
    change = followup_volume_mm3 - baseline_volume_mm3
    percentage = 100.0 * change / baseline_volume_mm3
    annualized_atrophy = -percentage * 365.25 / interval_days
    return LongitudinalVolumeChange(
        baseline_volume_mm3=float(baseline_volume_mm3),
        followup_volume_mm3=float(followup_volume_mm3),
        interval_days=float(interval_days),
        absolute_change_mm3=float(change),
        percentage_change=float(percentage),
        annualized_atrophy_percent=float(annualized_atrophy),
    )


def _label_volume(segmentation: np.ndarray, labels: tuple[int, ...], voxel_volume: float) -> float:
    return float(np.count_nonzero(np.isin(segmentation, labels)) * voxel_volume)
