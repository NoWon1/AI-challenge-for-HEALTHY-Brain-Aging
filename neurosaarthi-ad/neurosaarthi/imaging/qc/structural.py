"""Deterministic structural-MRI QC checks.

These checks identify obvious engineering failures. They are not a substitute
for secure visual review by qualified researchers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from neurosaarthi.imaging.io.nifti import ImageVolume


class QCSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class QCFlag:
    code: str
    severity: QCSeverity
    message: str


@dataclass(frozen=True)
class QCConfig:
    min_dimension: int = 32
    max_dimension: int = 512
    min_spacing_mm: float = 0.35
    max_spacing_mm: float = 5.0
    min_fov_mm: float = 80.0
    max_fov_mm: float = 420.0
    min_foreground_fraction: float = 0.02
    max_foreground_fraction: float = 0.95
    max_empty_internal_slice_fraction: float = 0.08
    max_slice_discontinuity: float = 2.5


@dataclass
class QCReport:
    shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    foreground_fraction: float
    finite_fraction: float
    intensity_mean: float
    intensity_std: float
    slice_discontinuity: float
    flags: list[QCFlag] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(flag.severity is QCSeverity.ERROR for flag in self.flags)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def estimate_foreground_mask(data: np.ndarray) -> np.ndarray:
    """Estimate non-background voxels using a robust low-intensity threshold."""

    array = np.asarray(data, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=bool)
    nonzero = np.abs(finite[np.abs(finite) > np.finfo(float).eps])
    if nonzero.size == 0:
        return np.zeros(array.shape, dtype=bool)
    threshold = max(float(np.percentile(nonzero, 1.0)) * 0.25, np.finfo(float).eps)
    return np.isfinite(array) & (np.abs(array) > threshold)


def run_structural_qc(volume: ImageVolume, config: QCConfig | None = None) -> QCReport:
    """Run geometry, finite-value, foreground, and slice-consistency checks."""

    active = config or QCConfig()
    data = np.asarray(volume.data, dtype=float)
    spacing = volume.voxel_spacing_mm
    finite_mask = np.isfinite(data)
    finite_fraction = float(finite_mask.mean())
    foreground = estimate_foreground_mask(data)
    foreground_fraction = float(foreground.mean())
    foreground_values = data[foreground & finite_mask]
    mean = float(np.mean(foreground_values)) if foreground_values.size else float("nan")
    std = float(np.std(foreground_values)) if foreground_values.size else float("nan")
    discontinuity = _slice_discontinuity(data, finite_mask, std)
    flags: list[QCFlag] = []

    if any(size < active.min_dimension or size > active.max_dimension for size in data.shape):
        flags.append(
            QCFlag(
                "unexpected_dimensions", QCSeverity.ERROR, "Image dimensions are outside configured bounds"
            )
        )
    if finite_fraction < 1.0:
        flags.append(QCFlag("non_finite", QCSeverity.ERROR, "Image contains NaN or infinite voxels"))
    if np.linalg.matrix_rank(volume.affine[:3, :3]) < 3 or not np.allclose(volume.affine[3], [0, 0, 0, 1]):
        flags.append(
            QCFlag("invalid_affine", QCSeverity.ERROR, "Affine is singular or has an invalid homogeneous row")
        )
    if any(item < active.min_spacing_mm or item > active.max_spacing_mm for item in spacing):
        flags.append(
            QCFlag("unexpected_spacing", QCSeverity.ERROR, "Voxel spacing is outside configured bounds")
        )
    fov = tuple(size * step for size, step in zip(data.shape, spacing, strict=True))
    if any(item < active.min_fov_mm or item > active.max_fov_mm for item in fov):
        flags.append(
            QCFlag(
                "unexpected_fov", QCSeverity.WARNING, "Physical field of view is outside configured bounds"
            )
        )
    if foreground_fraction < active.min_foreground_fraction:
        flags.append(QCFlag("low_foreground", QCSeverity.ERROR, "Foreground fraction is too low"))
    if foreground_fraction > active.max_foreground_fraction:
        flags.append(
            QCFlag("high_foreground", QCSeverity.WARNING, "Foreground fraction is unexpectedly high")
        )
    if foreground_values.size == 0 or not np.isfinite(std) or std <= np.finfo(float).eps:
        flags.append(
            QCFlag("constant_intensity", QCSeverity.ERROR, "Foreground has no usable intensity variation")
        )
    if _empty_internal_slice_fraction(foreground) > active.max_empty_internal_slice_fraction:
        flags.append(
            QCFlag(
                "missing_slices",
                QCSeverity.ERROR,
                "Unexpected empty slices occur inside the foreground extent",
            )
        )
    if discontinuity > active.max_slice_discontinuity:
        flags.append(
            QCFlag(
                "slice_discontinuity", QCSeverity.WARNING, "Adjacent-slice intensity discontinuity is high"
            )
        )

    return QCReport(
        shape=tuple(int(item) for item in data.shape),
        voxel_spacing_mm=spacing,
        foreground_fraction=foreground_fraction,
        finite_fraction=finite_fraction,
        intensity_mean=mean,
        intensity_std=std,
        slice_discontinuity=discontinuity,
        flags=flags,
    )


def _empty_internal_slice_fraction(mask: np.ndarray) -> float:
    worst = 0.0
    for axis in range(3):
        occupied = np.any(mask, axis=tuple(item for item in range(3) if item != axis))
        indices = np.flatnonzero(occupied)
        if indices.size < 2:
            continue
        internal = occupied[indices[0] : indices[-1] + 1]
        worst = max(worst, float(1.0 - internal.mean()))
    return worst


def _slice_discontinuity(data: np.ndarray, finite_mask: np.ndarray, foreground_std: float) -> float:
    if not np.isfinite(foreground_std) or foreground_std <= np.finfo(float).eps:
        return float("inf")
    safe = np.where(finite_mask, data, 0.0)
    scores = []
    for axis in range(3):
        first = np.take(safe, indices=range(1, safe.shape[axis]), axis=axis)
        second = np.take(safe, indices=range(0, safe.shape[axis] - 1), axis=axis)
        scores.append(float(np.mean(np.abs(first - second)) / foreground_std))
    return max(scores)
