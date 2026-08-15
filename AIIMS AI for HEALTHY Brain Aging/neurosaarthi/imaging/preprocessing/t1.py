"""A transparent T1 MRI preprocessing baseline.

The pipeline performs QC, optional N4 correction, spacing standardisation,
foreground-centred crop/pad, and robust foreground normalisation. Registration
and skull stripping remain separate, explicit stages rather than hidden side
effects of this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.imaging.io.nifti import ImageVolume
from neurosaarthi.imaging.qc.structural import QCConfig, QCReport, estimate_foreground_mask, run_structural_qc


@dataclass(frozen=True)
class T1PreprocessConfig:
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    output_shape: tuple[int, int, int] = (160, 192, 160)
    intensity_lower_percentile: float = 0.5
    intensity_upper_percentile: float = 99.5
    use_n4_bias_correction: bool = False
    fail_on_qc_error: bool = True

    def __post_init__(self) -> None:
        if any(item <= 0 for item in self.target_spacing_mm):
            raise DataValidationError("target_spacing_mm values must be positive")
        if any(item <= 0 for item in self.output_shape):
            raise DataValidationError("output_shape values must be positive")
        if not 0 <= self.intensity_lower_percentile < self.intensity_upper_percentile <= 100:
            raise DataValidationError("Intensity percentiles must satisfy 0 <= lower < upper <= 100")


@dataclass(frozen=True)
class SpatialTransformRecord:
    canonical_shape: tuple[int, int, int]
    native_shape: tuple[int, int, int]
    native_affine: np.ndarray
    canonical_to_native_ornt: np.ndarray | None
    resampled_shape: tuple[int, int, int]
    crop_start: tuple[int, int, int]
    crop_stop: tuple[int, int, int]
    pad_before: tuple[int, int, int]
    pad_after: tuple[int, int, int]

    def inverse_mask(self, processed_mask: np.ndarray) -> np.ndarray:
        """Map a processed-space discrete mask back to native image geometry."""

        mask = np.asarray(processed_mask)
        expected = tuple(
            stop - start + before + after
            for start, stop, before, after in zip(
                self.crop_start, self.crop_stop, self.pad_before, self.pad_after, strict=True
            )
        )
        if mask.shape != expected:
            raise DataValidationError(
                f"Processed mask shape {mask.shape} does not match expected shape {expected}"
            )
        unpadded_slices = tuple(
            slice(before, size - after if after else size)
            for before, after, size in zip(self.pad_before, self.pad_after, mask.shape, strict=True)
        )
        cropped = mask[unpadded_slices]
        resampled = np.zeros(self.resampled_shape, dtype=mask.dtype)
        target_slices = tuple(
            slice(start, stop) for start, stop in zip(self.crop_start, self.crop_stop, strict=True)
        )
        resampled[target_slices] = cropped
        canonical = _resize_nearest(resampled, self.canonical_shape)
        if self.canonical_to_native_ornt is None:
            native = canonical
        else:
            try:
                import nibabel as nib
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Native orientation inversion requires nibabel") from exc
            native = nib.orientations.apply_orientation(canonical, self.canonical_to_native_ornt)
        if native.shape != self.native_shape:
            native = _center_crop_or_pad(native, self.native_shape)
        return np.asarray(native, dtype=mask.dtype)


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    foreground_mask: np.ndarray
    affine: np.ndarray
    transform: SpatialTransformRecord
    input_qc: QCReport
    output_qc: QCReport


class T1Preprocessor:
    def __init__(self, config: T1PreprocessConfig | None = None, qc_config: QCConfig | None = None):
        self.config = config or T1PreprocessConfig()
        self.qc_config = qc_config or QCConfig()

    def preprocess(self, volume: ImageVolume) -> PreprocessResult:
        input_qc = run_structural_qc(volume, self.qc_config)
        if self.config.fail_on_qc_error and not input_qc.passed:
            codes = ", ".join(flag.code for flag in input_qc.flags if flag.severity.value == "error")
            raise DataValidationError(f"T1 MRI failed input QC: {codes}")

        data = np.asarray(volume.data, dtype=np.float32)
        foreground = estimate_foreground_mask(data)
        if self.config.use_n4_bias_correction:
            data = _n4_bias_correct(data, foreground, volume.voxel_spacing_mm)
        resampled, resampled_affine = _resample(
            data, volume.affine, volume.voxel_spacing_mm, self.config.target_spacing_mm, order=1
        )
        resampled_mask, _ = _resample(
            foreground.astype(np.uint8),
            volume.affine,
            volume.voxel_spacing_mm,
            self.config.target_spacing_mm,
            order=0,
        )
        resampled_mask = resampled_mask > 0
        cropped, cropped_mask, crop_start, crop_stop, pad_before, pad_after = _crop_pad_foreground(
            resampled, resampled_mask, self.config.output_shape
        )
        processed_affine = resampled_affine.copy()
        voxel_offset = np.asarray(crop_start, dtype=float) - np.asarray(pad_before, dtype=float)
        processed_affine[:3, 3] = resampled_affine[:3, 3] + resampled_affine[:3, :3] @ voxel_offset
        normalized = _robust_normalize(
            cropped,
            cropped_mask,
            self.config.intensity_lower_percentile,
            self.config.intensity_upper_percentile,
        )
        processed_volume = ImageVolume(normalized, processed_affine)
        output_qc = run_structural_qc(processed_volume, self.qc_config)
        transform = SpatialTransformRecord(
            canonical_shape=tuple(int(item) for item in volume.data.shape),
            native_shape=volume.native_shape or tuple(int(item) for item in volume.data.shape),
            native_affine=np.asarray(volume.native_affine, dtype=float),
            canonical_to_native_ornt=volume.canonical_to_native_ornt,
            resampled_shape=tuple(int(item) for item in resampled.shape),
            crop_start=crop_start,
            crop_stop=crop_stop,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        return PreprocessResult(
            image=normalized.astype(np.float32),
            foreground_mask=cropped_mask.astype(bool),
            affine=processed_affine,
            transform=transform,
            input_qc=input_qc,
            output_qc=output_qc,
        )


def _resample(
    data: np.ndarray,
    affine: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(source_spacing) / np.asarray(target_spacing)
    if np.allclose(factors, 1.0):
        return np.asarray(data).copy(), np.asarray(affine, dtype=float).copy()
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Spacing resampling requires scipy from the imaging extra") from exc
    result = zoom(data, zoom=factors, order=order, mode="nearest", prefilter=order > 1)
    new_affine = np.asarray(affine, dtype=float).copy()
    directions = new_affine[:3, :3] / np.asarray(source_spacing)[None, :]
    new_affine[:3, :3] = directions * np.asarray(target_spacing)[None, :]
    return np.asarray(result), new_affine


def _crop_pad_foreground(
    data: np.ndarray, mask: np.ndarray, output_shape: tuple[int, int, int]
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    if mask.any():
        center = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    else:
        center = np.asarray(data.shape) // 2
    start = np.maximum(0, center - np.asarray(output_shape) // 2)
    stop = np.minimum(np.asarray(data.shape), start + np.asarray(output_shape))
    start = np.maximum(0, stop - np.asarray(output_shape))
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop, strict=True))
    cropped = data[slices]
    cropped_mask = mask[slices]
    missing = np.asarray(output_shape) - np.asarray(cropped.shape)
    before = np.maximum(0, missing // 2)
    after = np.maximum(0, missing - before)
    padding = tuple((int(left), int(right)) for left, right in zip(before, after, strict=True))
    return (
        np.pad(cropped, padding, mode="constant"),
        np.pad(cropped_mask, padding, mode="constant"),
        tuple(int(item) for item in start),
        tuple(int(item) for item in stop),
        tuple(int(item) for item in before),
        tuple(int(item) for item in after),
    )


def _robust_normalize(data: np.ndarray, mask: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(data, dtype=float)[mask]
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise DataValidationError("Cannot normalize an image without foreground intensity variation")
    low, high = np.percentile(values, [lower, upper])
    if high <= low:
        raise DataValidationError("Robust intensity range is degenerate")
    clipped = np.clip(np.asarray(data, dtype=float), low, high)
    clipped_values = clipped[mask]
    mean = float(np.mean(clipped_values))
    std = float(np.std(clipped_values))
    if std <= np.finfo(float).eps:
        raise DataValidationError("Foreground intensity standard deviation is zero")
    normalized = (clipped - mean) / std
    normalized[~mask] = 0.0
    return normalized


def _n4_bias_correct(data: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("N4 bias correction requires SimpleITK from the imaging extra") from exc
    image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)).astype(np.float32))
    image.SetSpacing(tuple(float(item) for item in spacing))
    mask_image = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)).astype(np.uint8))
    mask_image.CopyInformation(image)
    corrected = sitk.N4BiasFieldCorrectionImageFilter().Execute(image, mask_image)
    return np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0)).astype(np.float32)


def _resize_nearest(data: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if data.shape == target_shape:
        return data.copy()
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Mask inversion requires scipy from the imaging extra") from exc
    factors = np.asarray(target_shape) / np.asarray(data.shape)
    resized = zoom(data, zoom=factors, order=0, mode="nearest", prefilter=False)
    return _center_crop_or_pad(resized, target_shape)


def _center_crop_or_pad(data: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    slices = []
    padding = []
    for size, target in zip(data.shape, target_shape, strict=True):
        if size > target:
            start = (size - target) // 2
            slices.append(slice(start, start + target))
            padding.append((0, 0))
        else:
            slices.append(slice(0, size))
            missing = target - size
            padding.append((missing // 2, missing - missing // 2))
    return np.pad(data[tuple(slices)], tuple(padding), mode="constant")
