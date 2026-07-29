"""
V-Net configuration with hardcoded parameters.

All architecture and preprocessing parameters are directly embedded in
this file for the bronchial artery CTA segmentation task.

No external configuration files are required at runtime.

Dataset: xiangxi_bronchial_artery_cta
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IntensityProperties:
    """Foreground intensity statistics of the dataset."""
    max: float
    mean: float
    median: float
    min: float
    percentile_00_5: float
    percentile_99_5: float
    std: float


@dataclass
class ArchitectureConfig:
    """V-Net architecture parameters."""
    n_stages: int
    features_per_stage: List[int]
    kernel_sizes: List[List[int]]
    strides: List[List[int]]
    n_conv_per_stage: List[int]
    n_conv_per_stage_decoder: List[int]
    conv_bias: bool
    norm_op: str
    norm_op_kwargs: dict
    nonlin: str
    nonlin_kwargs: dict
    conv_op: str


@dataclass
class PlanConfig:
    """Full configuration for one cascade stage (e.g. 3d_fullres)."""
    config_name: str
    preprocessor_name: str
    batch_size: int
    patch_size: List[int]
    median_image_size_in_voxels: List[float]
    spacing: List[float]
    normalization_schemes: List[str]
    use_mask_for_norm: List[bool]
    architecture: ArchitectureConfig
    batch_dice: bool
    next_stage: Optional[str] = None
    previous_stage: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset-level configuration."""
    dataset_name: str
    config_scheme: str
    original_median_spacing: List[float]
    original_median_shape: List[int]
    image_reader_writer: str
    transpose_forward: List[int]
    transpose_backward: List[int]
    intensity_properties: IntensityProperties


# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

# --- Intensity properties (foreground statistics) ---
_INTENSITY = IntensityProperties(
    max=828.0,
    mean=145.37766868933042,
    median=134.0,
    min=-1006.0,
    percentile_00_5=-207.0,
    percentile_99_5=511.0,
    std=107.59070675735042,
)

# --- Shared V-Net architecture (identical for coarse and fine stages) ---
_ARCH = ArchitectureConfig(
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[[3, 3, 3]] * 6,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
    conv_bias=True,
    norm_op="InstanceNorm3d",
    norm_op_kwargs={"eps": 1e-5, "affine": True},
    nonlin="LeakyReLU",
    nonlin_kwargs={"inplace": True},
    conv_op="Conv3d",
)

# --- Coarse stage (low resolution) ---
LOWRES_CONFIG = PlanConfig(
    config_name="vnet_3d_lowres",
    preprocessor_name="StandardPreprocessor",
    batch_size=2,
    patch_size=[128, 128, 128],
    median_image_size_in_voxels=[187.0, 211.0, 211.0],
    spacing=[3.0, 3.0, 3.0],
    normalization_schemes=["ZScoreNormalization"],
    use_mask_for_norm=[False],
    architecture=_ARCH,
    batch_dice=False,
    next_stage="vnet_3d_cascade_fullres",
)

# --- Fine stage (full resolution) ---
FULLRES_CONFIG = PlanConfig(
    config_name="vnet_3d_fullres",
    preprocessor_name="StandardPreprocessor",
    batch_size=2,
    patch_size=[128, 128, 128],
    median_image_size_in_voxels=[454.5, 512.0, 512.0],
    spacing=[1.0, 1.0, 1.0],
    normalization_schemes=["ZScoreNormalization"],
    use_mask_for_norm=[False],
    architecture=_ARCH,
    batch_dice=True,
)

# --- Dataset-level config ---
DATASET_CONFIG = DatasetConfig(
    dataset_name="xiangxi_bronchial_artery_cta",
    config_scheme="VNetPlans",
    original_median_spacing=[1.0, 1.0, 1.0],
    original_median_shape=[452, 512, 512],
    image_reader_writer="SimpleITKIO",
    transpose_forward=[0, 1, 2],
    transpose_backward=[0, 1, 2],
    intensity_properties=_INTENSITY,
)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def get_default_configs() -> Tuple[DatasetConfig, PlanConfig, PlanConfig]:
    """
    Return dataset config plus the two cascade stages (lowres + fullres).

    All parameters are hardcoded; no external files are needed.
    """
    return DATASET_CONFIG, LOWRES_CONFIG, FULLRES_CONFIG


def get_plan_config(dataset_config: DatasetConfig, config_name: str) -> PlanConfig:
    """Return a specific plan configuration by name."""
    if config_name in ("3d_lowres", "lowres"):
        return LOWRES_CONFIG
    elif config_name in ("3d_fullres", "fullres"):
        return FULLRES_CONFIG
    elif config_name in ("3d_cascade_fullres", "cascade"):
        cfg = PlanConfig(
            config_name="vnet_3d_cascade_fullres",
            preprocessor_name=FULLRES_CONFIG.preprocessor_name,
            batch_size=FULLRES_CONFIG.batch_size,
            patch_size=FULLRES_CONFIG.patch_size,
            median_image_size_in_voxels=FULLRES_CONFIG.median_image_size_in_voxels,
            spacing=FULLRES_CONFIG.spacing,
            normalization_schemes=FULLRES_CONFIG.normalization_schemes,
            use_mask_for_norm=FULLRES_CONFIG.use_mask_for_norm,
            architecture=FULLRES_CONFIG.architecture,
            batch_dice=FULLRES_CONFIG.batch_dice,
            previous_stage="3d_lowres",
        )
        return cfg
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
