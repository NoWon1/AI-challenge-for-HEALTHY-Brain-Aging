"""NIfTI loading and saving without implicit network or identifier handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurosaarthi.core.errors import DataValidationError


@dataclass(frozen=True)
class ImageVolume:
    """In-memory 3D image and the geometry required for native-space export."""

    data: np.ndarray
    affine: np.ndarray
    native_shape: tuple[int, int, int] | None = None
    native_affine: np.ndarray | None = None
    canonical_to_native_ornt: np.ndarray | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        affine = np.asarray(self.affine, dtype=float)
        if data.ndim != 3:
            raise DataValidationError(f"Expected a 3D image, received shape={data.shape}")
        if any(size <= 0 for size in data.shape):
            raise DataValidationError("Image dimensions must be positive")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise DataValidationError("Image affine must be a finite 4x4 matrix")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "affine", affine)
        if self.native_shape is None:
            object.__setattr__(self, "native_shape", tuple(int(item) for item in data.shape))
        if self.native_affine is None:
            object.__setattr__(self, "native_affine", affine.copy())

    @property
    def voxel_spacing_mm(self) -> tuple[float, float, float]:
        spacing = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
        return tuple(float(item) for item in spacing)


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NIfTI I/O requires the 'imaging' optional dependencies") from exc
    return nib


def load_nifti(path: str | Path, *, canonical: bool = True, dtype: np.dtype = np.float32) -> ImageVolume:
    """Load a local NIfTI image, optionally orienting it to closest canonical."""

    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"NIfTI image not found: {image_path}")
    if not (image_path.name.lower().endswith(".nii") or image_path.name.lower().endswith(".nii.gz")):
        raise DataValidationError("Only .nii and .nii.gz inputs are accepted")
    nib = _require_nibabel()
    native = nib.load(str(image_path))
    if len(native.shape) != 3:
        raise DataValidationError(f"Expected a 3D NIfTI image, received shape={native.shape}")
    selected = nib.as_closest_canonical(native) if canonical else native
    orientation = None
    if canonical:
        native_ornt = nib.orientations.io_orientation(native.affine)
        selected_ornt = nib.orientations.io_orientation(selected.affine)
        orientation = nib.orientations.ornt_transform(selected_ornt, native_ornt)
    return ImageVolume(
        data=np.asarray(selected.dataobj, dtype=dtype),
        affine=np.asarray(selected.affine, dtype=float),
        native_shape=tuple(int(item) for item in native.shape),
        native_affine=np.asarray(native.affine, dtype=float),
        canonical_to_native_ornt=orientation,
    )


def save_nifti(
    data: np.ndarray, affine: np.ndarray, path: str | Path, *, dtype: np.dtype | None = None
) -> Path:
    """Save a 3D array to a local NIfTI path."""

    output_path = Path(path).resolve()
    if not (output_path.name.lower().endswith(".nii") or output_path.name.lower().endswith(".nii.gz")):
        raise DataValidationError("NIfTI output must end with .nii or .nii.gz")
    array = np.asarray(data, dtype=dtype)
    if array.ndim != 3:
        raise DataValidationError("Only 3D NIfTI outputs are supported")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib = _require_nibabel()
    nib.save(nib.Nifti1Image(array, np.asarray(affine, dtype=float)), str(output_path))
    return output_path
