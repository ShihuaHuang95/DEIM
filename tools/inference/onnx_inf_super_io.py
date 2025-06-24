# ruff: noqa: E402 F401
import argparse
import json
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cupy
except Exception:
    ModuleNotFoundError(
        "Ensure cupy is installed = ─ uv pip install cupy-cuda12x or cupy-cuda11x check nvidia-smi"
    )


# Import from our annotation package
from annotate import (
    DetectionResult,
    ResizeInfo,
    annotate_batch,
    annotate_detections,
    annotate_frames,
    apply_detection_preprocessing,
    apply_detection_preprocessing_batch,
    resize_batch_with_aspect_ratio,
    resize_with_aspect_ratio,
)


def check_dependencies():
    """Verify all required dependencies are installed."""
    missing_deps = []

    # Essential imports with proper error handling
    try:
        from loguru import logger
    except ImportError:
        missing_deps.append("loguru")
        # Since we can't use logger yet, print the error
        print(
            "Error: loguru package is missing. Please install it with 'pip install loguru'"
        )
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")

    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import supervision as sv
    except ImportError:
        missing_deps.append("supervision")

    # Check CuPy
    try:
        import cupy as cp
    except ImportError as e:
        missing_deps.append(
            "cupy-cuda11x (or appropriate version for your CUDA eg cupy-cuda12x)"
        )
        print(f"Error importing CuPy: {e}")
        print("Please ensure CuPy is installed correctly for your CUDA version.")
        print("e.g., pip install cupy-cuda11x (for CUDA 11.x)")

    # Check ONNX Runtime
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            # Cannot use logger yet if loguru is missing
            if "loguru" not in missing_deps:
                from loguru import logger

                logger.warning(
                    "onnxruntime is installed but CUDAExecutionProvider is not available"
                )
                logger.warning("Consider installing onnxruntime-gpu instead")
            else:
                print(
                    "Warning: onnxruntime is installed but CUDAExecutionProvider is not available"
                )
                print("Warning: Consider installing onnxruntime-gpu instead")
            missing_deps.append("onnxruntime-gpu")
    except ImportError as e:
        missing_deps.append("onnxruntime-gpu")
        print(f"Error importing onnxruntime: {e}")
        print("Please ensure onnxruntime-gpu is installed.")
        print("e.g., pip install onnxruntime-gpu")

    # If any dependencies are missing, exit with error
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install required packages and try again")
        sys.exit(1)

    return True


# Check dependencies early
if not check_dependencies():
    sys.exit(1)

# Now import dependencies safely
import cupy as cp
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

# =============================================================================
# Configuration and Constants
# =============================================================================

# Default model input size, adjust as needed
DEFAULT_MODEL_INPUT_SIZE = 640

# Valid range for configuration parameters
CONFIG_RANGES = {
    "batch_size": (1, 512),  # Min and max batch size
    "conf_threshold": (0.01, 1.0),  # Min and max confidence threshold
    "model_size": (64, 4096),  # Min and max model input size
    "cuda_device_id": (0, 16),  # Min and max CUDA device ID
}

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class InferenceStats:
    """Stores timing statistics for inference tracking."""

    total_read: float = 0
    total_preprocess: float = 0
    total_stack_transfer: float = 0
    total_binding_prep: float = 0
    total_inference: float = 0
    total_gpu_postprocess: float = 0
    total_annotate: float = 0
    total_write: float = 0
    total_loop: float = 0

    def __str__(self) -> str:
        """Pretty print the inference statistics."""
        return (
            f"Read:                    {self.total_read:.6f}s\n"
            f"Preprocess:              {self.total_preprocess:.6f}s\n"
            f"Stack & Transfer:        {self.total_stack_transfer:.6f}s\n"
            f"IO Binding Prep:         {self.total_binding_prep:.6f}s\n"
            f"Inference:               {self.total_inference:.6f}s\n"
            f"GPU Post-Process:        {self.total_gpu_postprocess:.6f}s\n"
            f"Annotation:              {self.total_annotate:.6f}s\n"
            f"Write:                   {self.total_write:.6f}s\n"
        )

    def get_percentages(self, total_time: float) -> Dict[str, float]:
        """Calculate percentage of total time for each component."""
        if total_time <= 0:
            return {}

        return {
            "Read": 100 * self.total_read / total_time,
            "Preprocess": 100 * self.total_preprocess / total_time,
            "Stack & Transfer": 100 * self.total_stack_transfer / total_time,
            "IO Binding Prep": 100 * self.total_binding_prep / total_time,
            "Inference": 100 * self.total_inference / total_time,
            "GPU Post-Process": 100 * self.total_gpu_postprocess / total_time,
            "Annotation": 100 * self.total_annotate / total_time,
            "Write": 100 * self.total_write / total_time,
        }


@dataclass
class Config:
    """Configuration parameters for inference."""

    model_path: Path
    input_path: Path
    output_path: Path  # Now expects the full file path
    labels_dict: dict
    batch_size: int = 16
    conf_threshold: float = 0.5
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE
    cuda_device_id: int = 0
    debug: bool = False

    def post_init(self):
        """Validate configuration parameters after initialization."""
        self.validate()

    def validate(self):
        """Validate all configuration parameters are within valid ranges."""
        # Check batch size
        if not (
            CONFIG_RANGES["batch_size"][0]
            <= self.batch_size
            <= CONFIG_RANGES["batch_size"][1]
        ):
            raise ValueError(
                f"Batch size {self.batch_size} out of valid range "
                f"{CONFIG_RANGES['batch_size'][0]}-{CONFIG_RANGES['batch_size'][1]}"
            )

        # Check confidence threshold
        if not (
            CONFIG_RANGES["conf_threshold"][0]
            <= self.conf_threshold
            <= CONFIG_RANGES["conf_threshold"][1]
        ):
            raise ValueError(
                f"Confidence threshold {self.conf_threshold} out of valid range "
                f"{CONFIG_RANGES['conf_threshold'][0]}-{CONFIG_RANGES['conf_threshold'][1]}"
            )

        # Check model input size
        if not (
            CONFIG_RANGES["model_size"][0]
            <= self.model_input_size
            <= CONFIG_RANGES["model_size"][1]
        ):
            raise ValueError(
                f"Model input size {self.model_input_size} out of valid range "
                f"{CONFIG_RANGES['model_size'][0]}-{CONFIG_RANGES['model_size'][1]}"
            )

        # Check CUDA device ID
        if not (
            CONFIG_RANGES["cuda_device_id"][0]
            <= self.cuda_device_id
            <= CONFIG_RANGES["cuda_device_id"][1]
        ):
            raise ValueError(
                f"CUDA device ID {self.cuda_device_id} out of valid range "
                f"{CONFIG_RANGES['cuda_device_id'][0]}-{CONFIG_RANGES['cuda_device_id'][1]}"
            )

        # Check if paths exist (Paths should be resolved before creating Config)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")


@dataclass(frozen=True)
class GPUResizeInfo:
    """A GPU-friendly version of ResizeInfo that uses CuPy arrays instead of lists."""

    ratios_cp: cp.ndarray  # Shape: (batch_size,)
    pad_w_cp: cp.ndarray  # Shape: (batch_size,)
    pad_h_cp: cp.ndarray  # Shape: (batch_size,)
    orig_h_cp: cp.ndarray  # Shape: (batch_size,)
    orig_w_cp: cp.ndarray  # Shape: (batch_size,)
    batch_size: int

    @classmethod
    def from_resize_infos(cls, resize_infos: List[ResizeInfo]):
        """Create a GPUResizeInfo from a list of ResizeInfo objects."""
        batch_size = len(resize_infos)
        if batch_size == 0:
            return cls(
                ratios_cp=cp.zeros((0,), dtype=cp.float32),
                pad_w_cp=cp.zeros((0,), dtype=cp.int32),
                pad_h_cp=cp.zeros((0,), dtype=cp.int32),
                orig_h_cp=cp.zeros((0,), dtype=cp.int32),
                orig_w_cp=cp.zeros((0,), dtype=cp.int32),
                batch_size=0,
            )

        # Transfer in a single batch to reduce overhead
        ratios = [info.ratio for info in resize_infos]
        pad_w = [info.pad_w for info in resize_infos]
        pad_h = [info.pad_h for info in resize_infos]
        orig_h = [info.original_height for info in resize_infos]
        orig_w = [info.original_width for info in resize_infos]

        # Create arrays directly on GPU
        ratios_cp = cp.array(ratios, dtype=cp.float32)
        pad_w_cp = cp.array(pad_w, dtype=cp.int32)
        pad_h_cp = cp.array(pad_h, dtype=cp.int32)
        orig_h_cp = cp.array(orig_h, dtype=cp.int32)
        orig_w_cp = cp.array(orig_w, dtype=cp.int32)

        return cls(
            ratios_cp=ratios_cp,
            pad_w_cp=pad_w_cp,
            pad_h_cp=pad_h_cp,
            orig_h_cp=orig_h_cp,
            orig_w_cp=orig_w_cp,
            batch_size=batch_size,
        )


class GPUMemoryPool:
    """
    Manages GPU memory allocations to reduce overhead from repeated allocations/deallocations.
    Uses a simple caching mechanism to reuse buffers of the same shape and dtype.
    """

    def __init__(self, device_id=0):
        """
        Initialize the GPU memory pool.

        Args:
            device_id: CUDA device ID to use
        """
        self.device_id = device_id
        self.buffers = {}  # (shape, dtype) -> list of available buffers
        self.allocated_bytes = 0
        self.total_allocations = 0
        self.cache_hits = 0

    def get(self, shape, dtype):
        """
        Get a buffer of the specified shape and dtype, reusing if available.
        If not available, will try to find a larger buffer and slice it.
        """
        key = (tuple(shape), dtype)

        # Exact match - best case
        if key in self.buffers and self.buffers[key]:
            self.cache_hits += 1
            return self.buffers[key].pop()

        # Try to find a larger buffer of the same dtype that we can slice
        buffer_size = np.prod(shape)
        for buf_shape, buf_dtype in self.buffers:
            if buf_dtype == dtype and np.prod(buf_shape) >= buffer_size:
                if self.buffers[(buf_shape, buf_dtype)]:
                    larger_buffer = self.buffers[(buf_shape, buf_dtype)].pop()
                    sliced_buffer = larger_buffer.reshape(-1)[:buffer_size].reshape(
                        shape
                    )
                    return sliced_buffer

        # Otherwise, allocate new
        self.total_allocations += 1
        buffer = cp.empty(shape, dtype=dtype)
        self.allocated_bytes += buffer.nbytes
        return buffer

    def put(self, buffer):
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: CuPy array to return to the pool
        """
        if buffer is None:
            return

        key = (buffer.shape, buffer.dtype)
        if key not in self.buffers:
            self.buffers[key] = []

        # Add the buffer to the cache
        self.buffers[key].append(buffer)

    def put_many(self, buffers):
        """
        Return multiple buffers to the pool.

        Args:
            buffers: List of CuPy arrays to return to the pool
        """
        for buffer in buffers:
            self.put(buffer)

    def clear(self):
        """Clear all buffers in the pool."""
        self.buffers.clear()
        self.allocated_bytes = 0
        # Only free all blocks when explicitly requested
        cp.get_default_memory_pool().free_all_blocks()

    def get_stats(self):
        """
        Get memory pool statistics.

        Returns:
            dict: Dictionary of memory pool stats
        """
        total_cached = sum(
            sum(b.nbytes for b in buffers) for buffers in self.buffers.values()
        )
        return {
            "total_allocations": self.total_allocations,
            "cache_hits": self.cache_hits,
            "hit_rate": self.cache_hits / max(1, self.total_allocations) * 100,
            "total_allocated_bytes": self.allocated_bytes,
            "current_cached_bytes": total_cached,
            "buffer_types": len(self.buffers),
            "total_cached_buffers": sum(
                len(buffers) for buffers in self.buffers.values()
            ),
        }


class ONNXInferencer:
    """Handles ONNX model loading, IO Binding, and inference execution with memory pooling."""

    def __init__(self, model_path: Path, device_id: int = 0):
        """
        Initialize the ONNX Inferencer.

        Args:
            model_path: Path object to the ONNX model file
            device_id: CUDA device ID to use
            use_memory_pool: Whether to use GPU memory pooling
        """
        self.model_path = model_path
        self.device_id = device_id

        # Initialize memory pool if enabled
        self.memory_pool = GPUMemoryPool(device_id)

        # Call original initialization code
        self.session = self._create_inference_session()
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        self.output_dtypes_np = [
            get_numpy_dtype_from_ort_type(out.type) for out in self.output_details
        ]
        self.model_input_shape = self.input_details[0].shape
        logger.debug(
            f"Model expected input shape (first input): {self.model_input_shape}"
        )

    def run_inference_with_binding(self, io_binding: ort.IOBinding):
        """
        Executes inference using the prepared IO Binding.

        Args:
            io_binding: IO binding object prepared with input/output buffers
        """
        try:
            self.session.run_with_iobinding(io_binding)
            # Results are directly placed into the bound output CuPy arrays
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.error(traceback.format_exc())
            raise

    def _create_inference_session(self) -> ort.InferenceSession:
        """
        Creates and configures the ONNX Runtime Inference Session.

        Returns:
            ort.InferenceSession: Configured ONNX Runtime session
        """
        available_providers = ort.get_available_providers()
        logger.debug(f"Available ONNX Runtime providers: {available_providers}")

        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider not found. "
                "Ensure 'onnxruntime-gpu' is installed and CUDA drivers are configured."
            )

        providers_list = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": self.device_id,
                    # Add other CUDA options if needed
                },
            ),
            "CPUExecutionProvider",  # Fallback
        ]

        try:
            # Create session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Create session - use str(Path) for compatibility
            sess = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers_list,
            )

            logger.debug(f"Successfully loaded model '{self.model_path}'")
            logger.debug(f"Using providers: {sess.get_providers()}")

            if "CUDAExecutionProvider" not in sess.get_providers():
                logger.warning("Warning: Failed to initialize CUDAExecutionProvider!")
                logger.warning(
                    "Falling back to CPU execution, performance will be significantly lower."
                )
                # Option to raise error or continue with CPU
                # raise RuntimeError("CUDAExecutionProvider failed to initialize.")

            return sess
        except Exception as e:
            logger.error(f"Error loading ONNX model '{self.model_path}': {e}")
            logger.error(
                "Please ensure the model path is correct and the model is valid."
            )
            logger.error(traceback.format_exc())
            raise  # Re-raise the exception for proper error handling upstream

    def prepare_io_binding(
        self, inputs_gpu: List[cp.ndarray], outputs_gpu: List[cp.ndarray]
    ) -> ort.IOBinding:
        """
        Creates and binds input/output GPU buffers for inference.

        Args:
            inputs_gpu: List of input tensors on GPU
            outputs_gpu: List of output tensors on GPU

        Returns:
            ort.IOBinding: IO binding object for ONNX Runtime
        """
        io_binding = self.session.io_binding()

        # Bind Inputs
        for i, name in enumerate(self.input_names):
            io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=inputs_gpu[i].dtype,
                shape=inputs_gpu[i].shape,
                buffer_ptr=inputs_gpu[i].data.ptr,
            )

        # Bind Outputs
        for i, name in enumerate(self.output_names):
            io_binding.bind_output(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=outputs_gpu[i].dtype,
                shape=outputs_gpu[i].shape,
                buffer_ptr=outputs_gpu[i].data.ptr,
            )
        return io_binding

    def get_buffers_for_batch(self, batch_size: int):
        """
        Get input and output buffers for a specific batch size, using memory pool if enabled.

        Args:
            batch_size: Size of batch to allocate buffers for

        Returns:
            Tuple[List[cp.ndarray], List[cp.ndarray]]: Tuple of (input_buffers, output_buffers)
        """
        inputs_gpu = []
        outputs_gpu = []

        # Prepare input buffers
        for i, inp_info in enumerate(self.input_details):
            shape = list(inp_info.shape)

            # Replace dynamic batch dimension with specified batch size
            if isinstance(shape[0], str) or shape[0] is None or shape[0] < 0:
                shape[0] = batch_size

            # Determine numpy dtype
            dtype = get_numpy_dtype_from_ort_type(inp_info.type)

            # Get buffer from pool or allocate new
            buffer = self.memory_pool.get(shape, dtype)

            inputs_gpu.append(buffer)

        # Prepare output buffers
        for i, out_info in enumerate(self.output_details):
            shape = list(out_info.shape)

            # Replace dynamic batch dimension
            if isinstance(shape[0], str) or shape[0] is None or shape[0] < 0:
                shape[0] = batch_size

            # Use pre-calculated numpy dtype
            dtype = self.output_dtypes_np[i]

            # Get buffer from pool or allocate new
            buffer = self.memory_pool.get(shape, dtype)

            outputs_gpu.append(buffer)

        return inputs_gpu, outputs_gpu

    def release_buffers(self, inputs_gpu, outputs_gpu):
        """
        Release buffers back to memory pool if enabled.

        Args:
            inputs_gpu: List of input GPU buffers
            outputs_gpu: List of output GPU buffers
        """

        # Return buffers to the pool
        for buffer in inputs_gpu:
            self.memory_pool.put(buffer)

        for buffer in outputs_gpu:
            self.memory_pool.put(buffer)

    def clean_memory(self):
        """Clean up GPU memory."""

        # Log memory pool statistics
        stats = self.memory_pool.get_stats()
        logger.debug(f"Memory pool stats: {stats}")
        self.memory_pool.clear()


class VideoProcessor:
    def __init__(
        self,
        inferencer: ONNXInferencer,
        video_path: Path,
        output_path: Path,
        class_names: Dict[int, str] = {},  # Make optional with None default
        batch_size: int = 16,
        conf_threshold: float = 0.5,
        model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
    ):
        """
        Initialize the VideoProcessor.

        Args:
            inferencer: ONNX inferencer object
            video_path: Path object to input video
            output_path: Path object for output video
            class_names: dictionary mapping class IDs to names for annotation (optional)
            batch_size: Number of frames to process in a batch
            conf_threshold: Confidence threshold for detections
            model_input_size: Input size for the model
        """
        # Store parameters
        self.inferencer = inferencer
        self.video_path = video_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.model_input_size = model_input_size

        # Initialize class_names if not provided
        self.class_names = class_names

        # Initialize video capture
        self.cap = None
        self.out_writer = None
        self._setup_video_io()

        # Timing stats
        self.frame_count_processed = 0
        self.stats = InferenceStats()

        # Get pre-allocated buffers from the inferencer
        self.inputs_gpu, self.outputs_gpu = self.inferencer.get_buffers_for_batch(
            self.batch_size
        )
        logger.debug("GPU Buffers pre-allocated.")

    def _prepare_second_input_batch(self, current_batch_size: int) -> np.ndarray:
        """
        Prepares the second input batch (original target sizes).

        Args:
            current_batch_size: Number of items in the current batch

        Returns:
            np.ndarray: Second input tensor for the model
        """
        if len(self.inferencer.input_details) < 2:
            raise ValueError(
                "Model requires at least 2 inputs, but only has "
                f"{len(self.inferencer.input_details)}"
            )

        inp2_info = self.inferencer.input_details[1]
        inp2_dtype = get_numpy_dtype_from_ort_type(inp2_info.type)

        # Create data: [model_input_size, model_input_size] repeated for each item in batch
        # This represents the original target sizes for the model
        batch_input2_np = np.array(
            [[self.model_input_size, self.model_input_size]] * current_batch_size,
            dtype=inp2_dtype,
        )

        # Reshape to match the expected shape for the current batch size
        inp2_shape_template = list(inp2_info.shape)
        if (
            isinstance(inp2_shape_template[0], str)
            or inp2_shape_template[0] is None
            or inp2_shape_template[0] < 0
        ):
            inp2_shape_template[0] = current_batch_size
        else:
            inp2_shape_template[0] = current_batch_size  # Override fixed dim if needed

        try:
            return batch_input2_np.reshape(inp2_shape_template)
        except ValueError as e:
            logger.error(f"Failed to reshape second input batch: {e}")
            logger.error(f"Input shape template: {inp2_shape_template}")
            logger.error(f"Batch input shape: {batch_input2_np.shape}")
            logger.error(traceback.format_exc())
            raise

    def _process_batch(
        self,
        batch_input1_np: np.ndarray,
        resize_info_buffer: List[ResizeInfo],
        original_frames_buffer: List[np.ndarray],
        is_full_batch: bool = True,
    ):
        """
        Processes a single batch of frames with minimal CPU-GPU transfers.

        Args:
            batch_input1_np: Batch of preprocessed images as NumPy array (NCHW)
            resize_info_buffer: List of resize information per image
            original_frames_buffer: List of original RGB frames
            is_full_batch: Whether this is a full batch using pre-allocated buffers
        """
        current_batch_size = len(original_frames_buffer)
        if current_batch_size == 0:
            return

        # --- Prepare Second Input & Transfer to GPU ---
        with timing_context(self.stats, "total_stack_transfer"):
            batch_input2_np = self._prepare_second_input_batch(current_batch_size)

            try:
                if is_full_batch:
                    # Use pre-allocated buffers
                    input1_gpu_target = self.inputs_gpu[0]
                    input2_gpu_target = self.inputs_gpu[1]
                    outputs_gpu_target = self.outputs_gpu
                    input1_gpu_target.set(batch_input1_np)
                    input2_gpu_target.set(batch_input2_np)
                else:
                    # Get temporary buffers for the smaller remaining batch
                    if hasattr(self.inferencer, "get_buffers_for_batch"):
                        # Use memory pool if available
                        temp_inputs, temp_outputs = (
                            self.inferencer.get_buffers_for_batch(current_batch_size)
                        )
                        input1_gpu_target = temp_inputs[0]
                        input2_gpu_target = temp_inputs[1]
                        outputs_gpu_target = temp_outputs

                        # Copy data to GPU
                        input1_gpu_target.set(batch_input1_np)
                        input2_gpu_target.set(batch_input2_np)
                    else:
                        # Fall back to original method
                        input1_gpu_target = cp.asarray(batch_input1_np)
                        input2_gpu_target = cp.asarray(batch_input2_np)

                        # Create temporary output buffers with correct smaller batch size
                        outputs_gpu_target = []
                        for i, out_info in enumerate(self.inferencer.output_details):
                            shape = list(out_info.shape)
                            if (
                                isinstance(shape[0], str)
                                or shape[0] is None
                                or shape[0] < 0
                            ):
                                shape[0] = current_batch_size
                            else:
                                shape[0] = current_batch_size  # Override fixed size
                            dtype = self.inferencer.output_dtypes_np[i]
                            outputs_gpu_target.append(cp.empty(shape, dtype=dtype))

                synchronize_gpu()  # Ensure transfers are complete
            except Exception as e:
                logger.error(f"Error transferring data to GPU: {e}")
                logger.error(traceback.format_exc())
                # Release temporary buffers if they were allocated
                if (
                    not is_full_batch
                    and hasattr(self.inferencer, "release_buffers")
                    and "temp_inputs" in locals()
                ):
                    self.inferencer.release_buffers(temp_inputs, temp_outputs)
                raise

        # --- Prepare IO Binding ---
        with timing_context(self.stats, "total_binding_prep"):
            try:
                # Pass the GPU buffers
                io_binding = self.inferencer.prepare_io_binding(
                    inputs_gpu=[input1_gpu_target, input2_gpu_target],
                    outputs_gpu=outputs_gpu_target,
                )
            except Exception as e:
                logger.error(f"Error preparing IO binding: {e}")
                logger.error(traceback.format_exc())
                # Release temporary buffers if they were allocated
                if (
                    not is_full_batch
                    and hasattr(self.inferencer, "release_buffers")
                    and "temp_inputs" in locals()
                ):
                    self.inferencer.release_buffers(temp_inputs, temp_outputs)
                raise

        # --- INFERENCE ---
        with timing_context(self.stats, "total_inference"):
            try:
                self.inferencer.run_inference_with_binding(io_binding)
                # Results are now in outputs_gpu_target
                synchronize_gpu()  # Ensure inference is complete
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                logger.error(traceback.format_exc())
                # Release temporary buffers if they were allocated
                if (
                    not is_full_batch
                    and hasattr(self.inferencer, "release_buffers")
                    and "temp_inputs" in locals()
                ):
                    self.inferencer.release_buffers(temp_inputs, temp_outputs)
                raise

        # --- PROCESS DETECTIONS ON GPU ---
        with timing_context(self.stats, "total_gpu_postprocess"):
            try:
                # Process detections using GPU optimization
                boxes_batch = []
                scores_batch = []
                labels_batch = []

                # Extract the inference results by converting from GPU to CPU
                for i in range(current_batch_size):
                    if isinstance(outputs_gpu_target[0], cp.ndarray):
                        boxes_batch.append(cp.asnumpy(outputs_gpu_target[1][i]))
                        scores_batch.append(cp.asnumpy(outputs_gpu_target[2][i]))
                        labels_batch.append(cp.asnumpy(outputs_gpu_target[0][i]))
                    else:
                        boxes_batch.append(outputs_gpu_target[1][i])
                        scores_batch.append(outputs_gpu_target[2][i])
                        labels_batch.append(outputs_gpu_target[0][i])

                # Process boxes to original image coordinates using annotate package
                adjusted_boxes_batch = apply_detection_preprocessing_batch(
                    boxes_batch=boxes_batch, resize_infos=resize_info_buffer
                )

                synchronize_gpu()  # Ensure GPU work is complete
            except Exception as e:
                logger.error(f"Error processing detections on GPU: {e}")
                logger.error(traceback.format_exc())
                # Release temporary buffers if they were allocated
                if (
                    not is_full_batch
                    and hasattr(self.inferencer, "release_buffers")
                    and "temp_inputs" in locals()
                ):
                    self.inferencer.release_buffers(temp_inputs, temp_outputs)
                raise

        # --- ANNOTATE & WRITE (CPU) ---
        with timing_context(self.stats, "total_annotate"):
            try:
                for labels in labels_batch:
                    unique_labels = set(labels.astype(int))
                    for label in unique_labels:
                        if label not in self.class_names:
                            self.class_names[label] = f"class_{label}"

                # Use annotate package to annotate frames
                annotated_frames = annotate_batch(
                    images=original_frames_buffer,
                    boxes_batch=adjusted_boxes_batch,
                    scores_batch=scores_batch,
                    labels_batch=labels_batch,
                    class_names=self.class_names,
                    conf_threshold=self.conf_threshold,
                )
            except Exception as e:
                logger.error(f"Error annotating frames: {e}")
                logger.error(traceback.format_exc())
                if (
                    not is_full_batch
                    and hasattr(self.inferencer, "release_buffers")
                    and "temp_inputs" in locals()
                ):
                    self.inferencer.release_buffers(temp_inputs, temp_outputs)
                raise

        with timing_context(self.stats, "total_write"):
            if self.out_writer:
                try:
                    for annotated_frame in annotated_frames:
                        # Convert back to BGR for VideoWriter
                        frame_out = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        self.out_writer.write(frame_out)
                except Exception as e:
                    logger.error(f"Error writing frames to output: {e}")
                    logger.error(traceback.format_exc())
                    raise

        # Release temporary buffers if not using pre-allocated
        if not is_full_batch:
            if (
                hasattr(self.inferencer, "release_buffers")
                and "temp_inputs" in locals()
            ):
                self.inferencer.release_buffers(temp_inputs, temp_outputs)
            else:
                # Original cleanup for non-memory pool case
                del input1_gpu_target, input2_gpu_target
                for buf in outputs_gpu_target:
                    del buf
                # Force memory release on partial batches
                cp.get_default_memory_pool().free_all_blocks()

    def run(self):
        """
        Runs the video processing loop with batch processing and proper resource management.
        """
        if self.cap is None or self.out_writer is None:
            logger.error("Video I/O not initialized properly")
            return

        total_start_time = time.time()

        # Buffers (CPU side)
        resize_info_buffer: List[ResizeInfo] = []
        original_frames_buffer: List[np.ndarray] = []  # Store original RGB frames
        frames_buffer_preprocessed_np: List[
            np.ndarray
        ] = []  # Store NCHW float32 numpy arrays

        try:
            while self.cap.isOpened():
                with timing_context(self.stats, "total_loop"):
                    # Read frame
                    with timing_context(self.stats, "total_read"):
                        ret, frame_bgr = self.cap.read()
                        if not ret:
                            break  # End of video

                    self.frame_count_processed += 1

                    with timing_context(self.stats, "total_preprocess"):
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        original_frames_buffer.append(frame_rgb)

                        # Use annotate.py package for resizing
                        resized_frame_rgb, resize_info = resize_with_aspect_ratio(
                            frame_rgb, self.model_input_size
                        )
                        resize_info_buffer.append(resize_info)

                        # Convert to numpy and normalize
                        if isinstance(resized_frame_rgb, cp.ndarray):
                            resized_frame_rgb_np = cp.asnumpy(resized_frame_rgb)
                        else:
                            resized_frame_rgb_np = resized_frame_rgb

                        # Normalize (0-1)
                        input_tensor_frame_np = (
                            resized_frame_rgb_np.astype(np.float32) / 255.0
                        )

                        # Transpose (HWC to CHW)
                        input_tensor_frame_np = input_tensor_frame_np.transpose(2, 0, 1)
                        frames_buffer_preprocessed_np.append(input_tensor_frame_np)

                    # --- Process in batches ---
                    if len(frames_buffer_preprocessed_np) == self.batch_size:
                        # Stack numpy arrays on CPU before passing to _process_batch
                        batch_input1_np = np.stack(frames_buffer_preprocessed_np)

                        self._process_batch(
                            batch_input1_np,
                            resize_info_buffer,
                            original_frames_buffer,
                            is_full_batch=True,
                        )

                        # Clear CPU buffers for next batch
                        frames_buffer_preprocessed_np.clear()
                        resize_info_buffer.clear()
                        original_frames_buffer.clear()

                    # Print progress periodically
                    if self.frame_count_processed % (self.batch_size * 5) == 0:
                        progress_pct = (
                            (self.frame_count_processed / self.frame_count_total * 100)
                            if self.frame_count_total > 0
                            else 0
                        )
                        logger.debug(
                            f"Processed {self.frame_count_processed}/{self.frame_count_total if self.frame_count_total > 0 else '?'} "
                            f"frames ({progress_pct:.1f}%)"
                        )

            # --- Process remaining frames ---
            remaining_count = len(frames_buffer_preprocessed_np)
            if remaining_count > 0:
                logger.debug(f"Processing {remaining_count} remaining frames...")
                batch_input1_np = np.stack(frames_buffer_preprocessed_np)
                self._process_batch(
                    batch_input1_np,
                    resize_info_buffer,
                    original_frames_buffer,
                    is_full_batch=False,  # Use temporary buffers
                )
                # Clear buffers
                frames_buffer_preprocessed_np.clear()
                resize_info_buffer.clear()
                original_frames_buffer.clear()

        except Exception:
            logger.error(traceback.format_exc())
            raise
        finally:
            # --- Cleanup and Stats ---
            total_time = time.time() - total_start_time

            # Properly release resources
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if self.out_writer is not None and self.out_writer.isOpened():
                self.out_writer.release()
            cv2.destroyAllWindows()

            # Print memory pool stats before cleaning up
            if hasattr(self.inferencer, "memory_pool") and self.inferencer.memory_pool:
                logger.debug(
                    f"Memory pool stats: {self.inferencer.memory_pool.get_stats()}"
                )

            # Clean up GPU memory through the inferencer
            self.inferencer.clean_memory()

            # Print stats if processing was successful
            if self.frame_count_processed > 0:
                self._print_statistics(total_time)
                logger.success(f"\nResult saved as '{self.output_path}'.")

    def _setup_video_io(self):
        """Setup video input and output objects with proper error handling."""
        try:
            # Use str(Path) for OpenCV compatibility
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise IOError(f"Could not open video {self.video_path}")

            # Video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Ensure output directory exists
            output_path = self.output_path.parent
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)

            # Create video writer - use str(Path) for OpenCV compatibility
            self.fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self.out_writer = cv2.VideoWriter(
                str(self.output_path), self.fourcc, self.fps, (self.orig_w, self.orig_h)
            )

            if not self.out_writer.isOpened():
                raise IOError(
                    f"Could not create output video writer for {self.output_path}"
                )

        except Exception:
            # Clean up resources on error
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if self.out_writer is not None and self.out_writer.isOpened():
                self.out_writer.release()
            raise

    def _print_statistics(self, total_time: float):
        """
        Prints timing statistics in a formatted way.

        Args:
            total_time: Total processing time in seconds
        """
        ml = multilogger("\n\n" + "=" * 50)

        ml.append("VIDEO PROCESSING COMPLETE (GPU-Accelerated IO Binding)")
        ml.append("=" * 50)

        if total_time > 0 and self.frame_count_processed > 0:
            avg_fps = self.frame_count_processed / total_time
            ml.append(
                f"Processed {self.frame_count_processed} frames in {total_time:.2f} seconds"
            )
            ml.append(f"Average FPS: {avg_fps:.2f}")

            fc = self.frame_count_processed  # Shorthand
            ml.append("\nAverage Time Per Frame Breakdown (seconds):")

            # Calculate per-frame stats
            per_frame_stats = {
                "Read": self.stats.total_read / fc,
                "Preprocess": self.stats.total_preprocess / fc,
                "Stack & Transfer": self.stats.total_stack_transfer / fc,
                "IO Binding Prep": self.stats.total_binding_prep / fc,
                "Inference": self.stats.total_inference / fc,
                "GPU Post-Process": self.stats.total_gpu_postprocess / fc,
                "Annotation": self.stats.total_annotate / fc,
                "Write": self.stats.total_write / fc,
            }

            # Print per-frame stats
            for key, value in per_frame_stats.items():
                ml.append(f"  {key.ljust(20)} {value:.6f}")
            ml.append("\n" * 2)

            # Calculate sum of tracked times, excluding loop time which encompasses everything
            sum_tracked = (
                self.stats.total_read
                + self.stats.total_preprocess
                + self.stats.total_stack_transfer
                + self.stats.total_binding_prep
                + self.stats.total_inference
                + self.stats.total_gpu_postprocess
                + self.stats.total_annotate
                + self.stats.total_write
            )
            other_time = total_time - sum_tracked  # Approximate other time

            # Get percentages
            percentages = self.stats.get_percentages(total_time)

            ml = ml.append("\nTotal Time Percentage Breakdown:")
            for key, value in percentages.items():
                ml.append(f"  {key.ljust(20)} {value:.1f}%")

            # Add the "Other/Overhead" percentage
            other_pct = 100 * other_time / total_time if total_time > 0 else 0
            ml.append(f"  {'Other/Overhead'.ljust(20)} {other_pct:.1f}%")

        else:
            logger.debug("No frames processed or total time was zero.")

        ml.append("=" * 50)
        ml.print("DEBUG")


class ImageProcessor:
    def __init__(self, inferencer: ONNXInferencer, class_names: Dict[int, str]):
        """
        Initialize the ImageProcessor.

        Args:
            inferencer: ONNX inferencer object
            class_names: dictionary mapping class IDs to names for annotation (optional)
        """
        self.inferencer = inferencer

        # Initialize class_names if not provided
        if class_names is None:
            class_names = {}
        self.class_names = class_names

        self.stats = InferenceStats()

    def process(
        self,
        img_path: Path,
        output_path: Path,
        conf_threshold: float = 0.5,
        model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
    ):
        """
        Processes a single image using the ONNXInferencer and IO Binding.

        Args:
            img_path: Path object to input image
            output_path: Path object for output image
            conf_threshold: Confidence threshold for detections
            model_input_size: Input size for the model

        Returns:
            bool: True if processing was successful, False otherwise
        """
        logger.debug(f"Processing single image: {img_path}")
        total_start_time = time.time()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read and validate input image
            with timing_context(self.stats, "total_read"):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.error(f"Could not read image {img_path}")
                    return False
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            logger.error(traceback.format_exc())
            return False

        # --- Preprocessing ---
        with timing_context(self.stats, "total_preprocess"):
            try:
                # Use annotate.py package for resizing
                resized_img, resize_info = resize_with_aspect_ratio(
                    img_rgb, model_input_size
                )

                # Convert to numpy and normalize
                if isinstance(resized_img, cp.ndarray):
                    resized_img_np = cp.asnumpy(resized_img)
                else:
                    resized_img_np = resized_img

                # Normalize (0-1)
                input_tensor_np = resized_img_np.astype(np.float32) / 255.0

                # Transpose (HWC to CHW)
                input_tensor_np = input_tensor_np.transpose(2, 0, 1)

                # Add batch dimension
                input_tensor_np = np.expand_dims(input_tensor_np, axis=0)

            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                logger.error(traceback.format_exc())
                return False

        # Prepare the second input (target size)
        if len(self.inferencer.input_details) < 2:
            logger.error(
                f"Model requires at least 2 inputs, but only has {len(self.inferencer.input_details)}"
            )
            return False

        inp2_info = self.inferencer.input_details[1]
        inp2_dtype = get_numpy_dtype_from_ort_type(inp2_info.type)
        inp2_shape = list(inp2_info.shape)

        # Replace dynamic dim with 1 (batch size)
        if isinstance(inp2_shape[0], str) or inp2_shape[0] is None or inp2_shape[0] < 0:
            inp2_shape[0] = 1

        input2_np = np.array([[model_input_size, model_input_size]], dtype=inp2_dtype)
        input2_np = input2_np.reshape(inp2_shape)  # Ensure exact shape

        # --- Prepare GPU Buffers & IO Binding ---
        with timing_context(self.stats, "total_binding_prep"):
            try:
                # Get buffers from the memory pool for batch size 1
                inputs_gpu, outputs_gpu = self.inferencer.get_buffers_for_batch(1)

                # Set the input data
                inputs_gpu[0].set(input_tensor_np)
                inputs_gpu[1].set(input2_np)

                # Create IO Binding using these buffers
                io_binding = self.inferencer.prepare_io_binding(
                    inputs_gpu=inputs_gpu, outputs_gpu=outputs_gpu
                )
            except Exception as e:
                logger.error(f"Error preparing GPU buffers and IO binding: {e}")
                logger.error(traceback.format_exc())
                # Release buffers if they were allocated
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Inference with IO Binding ---
        with timing_context(self.stats, "total_inference"):
            try:
                self.inferencer.run_inference_with_binding(io_binding)
                # Results are now in outputs_gpu
                synchronize_gpu()  # Ensure GPU work is done
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                logger.error(traceback.format_exc())
                # Release buffers
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Process detections ---
        with timing_context(self.stats, "total_gpu_postprocess"):
            try:
                # Extract results from GPU
                labels = (
                    cp.asnumpy(outputs_gpu[0][0])
                    if isinstance(outputs_gpu[0], cp.ndarray)
                    else outputs_gpu[0][0]
                )
                boxes = (
                    cp.asnumpy(outputs_gpu[1][0])
                    if isinstance(outputs_gpu[1], cp.ndarray)
                    else outputs_gpu[1][0]
                )
                scores = (
                    cp.asnumpy(outputs_gpu[2][0])
                    if isinstance(outputs_gpu[2], cp.ndarray)
                    else outputs_gpu[2][0]
                )

                # Process boxes to original image coordinates using annotate package
                adjusted_boxes = apply_detection_preprocessing(
                    boxes=boxes,
                    ratio=resize_info.ratio,
                    pad_w=resize_info.pad_w,
                    pad_h=resize_info.pad_h,
                    original_width=img_rgb.shape[1],
                    original_height=img_rgb.shape[0],
                )

                synchronize_gpu()  # Ensure post-processing is done
            except Exception as e:
                logger.error(f"Error processing detections: {e}")
                logger.error(traceback.format_exc())
                # Release buffers
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Annotate the image ---
        with timing_context(self.stats, "total_annotate"):
            try:
                # Use annotate.py package for annotation
                unique_labels = set(labels.astype(int))
                for label in unique_labels:
                    if label not in self.class_names:
                        self.class_names[label] = f"class_{label}"

                # Now call annotate_detections with the potentially updated class_names
                annotated_image = annotate_detections(
                    image=img_rgb,
                    boxes=adjusted_boxes,
                    scores=scores,
                    labels=labels,
                    class_names=self.class_names,
                    conf_threshold=conf_threshold,
                )
            except Exception as e:
                logger.error(f"Error annotating image: {e}")
                logger.error(traceback.format_exc())
                # Release buffers
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Save Result ---
        with timing_context(self.stats, "total_write"):
            try:
                result_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), result_bgr)
                logger.success(
                    f"Image processing complete. Result saved as '{output_path}'."
                )
            except Exception as e:
                logger.error(f"Error saving result image: {e}")
                logger.error(traceback.format_exc())
                return False
            finally:
                # Release buffers back to the memory pool
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)

        total_time = time.time() - total_start_time
        logger.debug(f"Total Image Processing Time: {total_time:.2f}s")

        # Print detailed timing breakdown
        logger.debug("\nTiming Breakdown:")
        for stat_name, stat_value in vars(self.stats).items():
            if stat_name.startswith("total_"):
                friendly_name = stat_name.replace("total_", "")
                logger.debug(
                    f"  {friendly_name.capitalize().ljust(15)}: {stat_value:.4f}s "
                    + (
                        f"({100 * stat_value / total_time:.1f}%)"
                        if total_time > 0
                        else ""
                    )
                )

        return True

    def process_batch(
        self,
        img_paths: List[Path],
        output_dir: Path,
        conf_threshold: float = 0.5,
        model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
    ):
        """
        Processes a batch of images using the ONNXInferencer and IO Binding.

        Args:
            img_paths: List of Path objects to input images
            output_dir: Path object for the output directory
            conf_threshold: Confidence threshold for detections
            model_input_size: Input size for the model

        Returns:
            bool: True if all processing was successful, False otherwise
        """
        logger.debug(f"Processing a batch of {len(img_paths)} images.")
        total_batch_start_time = time.time()
        all_successful = True

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Read and Preprocess Batch ---
        batch_size = len(img_paths)
        original_frames_buffer: List[np.ndarray] = []
        resize_info_buffer: List[ResizeInfo] = []
        frames_buffer_preprocessed_np: List[np.ndarray] = []

        with timing_context(self.stats, "total_preprocess"):
            for img_path in img_paths:
                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        logger.error(f"Could not read image {img_path}")
                        all_successful = False
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    original_frames_buffer.append(img_rgb)

                    # Use annotate.py package for resizing
                    resized_img, resize_info = resize_with_aspect_ratio(
                        img_rgb, model_input_size
                    )
                    resize_info_buffer.append(resize_info)

                    # Convert to numpy if necessary
                    if isinstance(resized_img, cp.ndarray):
                        resized_img_np = cp.asnumpy(resized_img)
                    else:
                        resized_img_np = resized_img

                    # Normalize (0-1)
                    input_tensor_np = resized_img_np.astype(np.float32) / 255.0

                    # Transpose (HWC to CHW)
                    input_tensor_np = input_tensor_np.transpose(2, 0, 1)
                    frames_buffer_preprocessed_np.append(input_tensor_np)

                except Exception as e:
                    logger.error(f"Error preprocessing image {img_path}: {e}")
                    logger.error(traceback.format_exc())
                    all_successful = False

            if not frames_buffer_preprocessed_np:
                logger.warning(
                    "No images were successfully preprocessed in this batch."
                )
                return False

            # Stack preprocessed images into a batch
            batch_input1_np = np.stack(frames_buffer_preprocessed_np)

        # Prepare the second input (target sizes)
        if len(self.inferencer.input_details) < 2:
            logger.error(
                f"Model requires at least 2 inputs, but only has {len(self.inferencer.input_details)}"
            )
            return False

        inp2_info = self.inferencer.input_details[1]
        inp2_dtype = get_numpy_dtype_from_ort_type(inp2_info.type)
        inp2_shape = list(inp2_info.shape)
        if isinstance(inp2_shape[0], str) or inp2_shape[0] is None or inp2_shape[0] < 0:
            inp2_shape[0] = batch_size
        input2_np = np.array(
            [[model_input_size, model_input_size]] * batch_size, dtype=inp2_dtype
        )
        input2_np = input2_np.reshape(inp2_shape)

        # --- Prepare GPU Buffers & IO Binding ---
        with timing_context(self.stats, "total_binding_prep"):
            try:
                inputs_gpu, outputs_gpu = self.inferencer.get_buffers_for_batch(
                    batch_size
                )
                inputs_gpu[0].set(batch_input1_np)
                inputs_gpu[1].set(input2_np)
                io_binding = self.inferencer.prepare_io_binding(
                    inputs_gpu=inputs_gpu, outputs_gpu=outputs_gpu
                )
            except Exception as e:
                logger.error(
                    f"Error preparing GPU buffers and IO binding for batch: {e}"
                )
                logger.error(traceback.format_exc())
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Inference with IO Binding ---
        with timing_context(self.stats, "total_inference"):
            try:
                self.inferencer.run_inference_with_binding(io_binding)
                synchronize_gpu()
            except Exception as e:
                logger.error(f"Error during batch inference: {e}")
                logger.error(traceback.format_exc())
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Process detections ---
        with timing_context(self.stats, "total_gpu_postprocess"):
            try:
                # Extract batch results from GPU
                boxes_batch = []
                scores_batch = []
                labels_batch = []

                # Convert from GPU to CPU if needed
                for i in range(batch_size):
                    if isinstance(outputs_gpu[1], cp.ndarray):
                        boxes_batch.append(cp.asnumpy(outputs_gpu[1][i]))
                        scores_batch.append(cp.asnumpy(outputs_gpu[2][i]))
                        labels_batch.append(cp.asnumpy(outputs_gpu[0][i]))
                    else:
                        boxes_batch.append(outputs_gpu[1][i])
                        scores_batch.append(outputs_gpu[2][i])
                        labels_batch.append(outputs_gpu[0][i])

                # Process boxes to original image coordinates using annotate package
                adjusted_boxes_batch = apply_detection_preprocessing_batch(
                    boxes_batch=boxes_batch, resize_infos=resize_info_buffer
                )

                synchronize_gpu()
            except Exception as e:
                logger.error(f"Error processing batch detections: {e}")
                logger.error(traceback.format_exc())
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        # --- Annotate and Save Results ---
        with timing_context(self.stats, "total_annotate"):
            try:
                # Use annotate.py package for batch annotation
                annotated_frames = annotate_batch(
                    images=original_frames_buffer,
                    boxes_batch=adjusted_boxes_batch,
                    scores_batch=scores_batch,
                    labels_batch=labels_batch,
                    class_names=self.class_names,
                    conf_threshold=conf_threshold,
                )
            except Exception as e:
                logger.error(f"Error annotating batch of images: {e}")
                logger.error(traceback.format_exc())
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)
                return False

        with timing_context(self.stats, "total_write"):
            try:
                for i, annotated_frame in enumerate(annotated_frames):
                    output_path = (
                        output_dir
                        / f"{Path(img_paths[i]).stem}_result{Path(img_paths[i]).suffix}"
                    )
                    result_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), result_bgr)
                    logger.success(f"Processed and saved: '{output_path}'")
            except Exception as e:
                logger.error(f"Error saving batch of result images: {e}")
                logger.error(traceback.format_exc())
                return False
            finally:
                self.inferencer.release_buffers(inputs_gpu, outputs_gpu)

        total_batch_time = time.time() - total_batch_start_time
        logger.debug(f"Total Batch Processing Time: {total_batch_time:.2f}s")

        return all_successful


# =============================================================================
# Utility Functions
# =============================================================================


class MultiLoggerHandler:
    """
    A handler for collecting multiple log entries and printing them as a group.
    This extends Loguru's functionality with a multi-line logging buffer.
    """

    def __init__(self, initial_value: str, prefix: str = ""):
        """Initialize with optional initial value and prefix for all entries"""
        self.entries: List[str] = []
        self.prefix = prefix
        if initial_value:
            self.entries.append(initial_value)

    def append(self, entry: Any) -> "MultiLoggerHandler":
        """Add an entry to the log buffer"""
        self.entries.append(f"{self.prefix}{entry}")
        return self

    def extend(self, entries: List[Any]) -> "MultiLoggerHandler":
        """Add multiple entries to the log buffer"""
        for entry in entries:
            self.append(entry)
        return self

    def print(self, level: str = "DEBUG") -> None:
        """Print all collected entries at the specified log level"""
        if not self.entries:
            return

        self.entries.append("\n")
        # Join all entries with newlines
        message = "\n".join(self.entries)

        # Log with the specified level
        getattr(logger, level.lower())(message)

        # Clear the entries after printing
        self.clear()

    def clear(self) -> "MultiLoggerHandler":
        """Clear all entries"""
        self.entries = []
        return self


# Extend Loguru's logger with multilogger method
def multilogger(initial_value: str, prefix: str = "") -> MultiLoggerHandler:
    """Create a new MultiLoggerHandler instance"""
    return MultiLoggerHandler(initial_value, prefix)


def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is an image based on its extension using pathlib.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if file is an image, False otherwise
    """
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def get_numpy_dtype_from_ort_type(ort_type_str: str) -> np.dtype:
    """
    Maps ONNX Runtime type string (e.g., 'tensor(float)') to numpy dtype.

    Args:
        ort_type_str: ONNX Runtime type string

    Returns:
        np.dtype: Corresponding NumPy data type
    """
    # create actual dtype objects
    type_map = {
        "float": np.dtype(np.float32),
        "float16": np.dtype(np.float16),
        "double": np.dtype(np.float64),
        "int8": np.dtype(np.int8),
        "int16": np.dtype(np.int16),
        "int32": np.dtype(np.int32),
        "int64": np.dtype(np.int64),
        "uint8": np.dtype(np.uint8),
        "uint16": np.dtype(np.uint16),
        "uint32": np.dtype(np.uint32),
        "uint64": np.dtype(np.uint64),
        "bool": np.dtype(np.bool_),
        "string": np.dtype(object),  # For tensors of strings
    }

    try:
        type_name = ort_type_str.split("(")[1].split(")")[0]
    except IndexError:
        raise ValueError(f"Could not parse ONNX Runtime type string: {ort_type_str}")

    if type_name in type_map:
        return type_map[type_name]
    else:
        raise NotImplementedError(
            f"Unsupported ONNX type string '{ort_type_str}' encountered. "
            f"Please update 'get_numpy_dtype_from_ort_type' mapping."
        )


@contextmanager
def timing_context(stat_obj: InferenceStats, stat_name: str):
    """
    Context manager for timing code blocks and updating stats.

    Args:
        stat_obj: Statistics object to update
        stat_name: Name of the statistic to update

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if hasattr(stat_obj, stat_name):
            setattr(stat_obj, stat_name, getattr(stat_obj, stat_name) + elapsed)


def synchronize_gpu(stream=None):
    """
    Synchronize the GPU to ensure operations are complete.

    Args:
        stream: CUDA stream to synchronize (default: None, uses default stream)
    """
    if stream is None:
        cp.cuda.Stream.null.synchronize()
    else:
        stream.synchronize()


def setup_cuda_device(device_id: int) -> bool:
    """
    Initialize the CUDA device safely with proper error handling.

    Args:
        device_id: CUDA device ID to use

    Returns:
        bool: True if device was set successfully, False otherwise
    """
    # Validate device ID is within range
    if (
        not CONFIG_RANGES["cuda_device_id"][0]
        <= device_id
        <= CONFIG_RANGES["cuda_device_id"][1]
    ):
        logger.error(
            f"CUDA device ID {device_id} out of valid range "
            f"{CONFIG_RANGES['cuda_device_id'][0]}-{CONFIG_RANGES['cuda_device_id'][1]}"
        )
        return False

    try:
        cp.cuda.Device(device_id).use()  # Set the default CuPy device
        logger.debug(f"Using CuPy with CUDA Device {device_id}")
        return True
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"Error setting CuPy device {device_id}: {e}")
        logger.error("Please check your CUDA installation and device ID.")
        logger.error(traceback.format_exc())
        return False


def process_detections_gpu(
    labels_batch_cp: Union[cp.ndarray, List],
    boxes_batch_cp: Union[cp.ndarray, List],
    scores_batch_cp: Union[cp.ndarray, List],
    resize_infos: List[ResizeInfo],
    threshold: float = 0.50,
) -> List[List[DetectionResult]]:
    """
    Process detection results for a batch of images and convert to DetectionResult list.
    This function is optimized for memory access and uses the annotate package.

    Args:
        labels_batch_cp: Batch of label tensors (CuPy or List)
        boxes_batch_cp: Batch of bounding box tensors (CuPy or List)
        scores_batch_cp: Batch of confidence score tensors (CuPy or List)
        resize_infos: List of resize information per image
        threshold: Confidence threshold for filtering detections

    Returns:
        List[List[DetectionResult]]: Processed detections per image
    """
    batch_size = len(resize_infos)
    if batch_size == 0:
        return []

    # Initialize result container
    processed_results = [[] for _ in range(batch_size)]

    # Process each image in the batch
    for i in range(batch_size):
        # Get data for this image
        labels = (
            labels_batch_cp[i]
            if isinstance(labels_batch_cp, list)
            else cp.asnumpy(labels_batch_cp[i])
        )
        boxes = (
            boxes_batch_cp[i]
            if isinstance(boxes_batch_cp, list)
            else cp.asnumpy(boxes_batch_cp[i])
        )
        scores = (
            scores_batch_cp[i]
            if isinstance(scores_batch_cp, list)
            else cp.asnumpy(scores_batch_cp[i])
        )

        # Apply threshold filter
        valid_indices = np.where(scores >= threshold)[0]

        for idx in valid_indices:
            detection = DetectionResult(
                box=boxes[idx], score=float(scores[idx]), label=int(labels[idx])
            )
            processed_results[i].append(detection)

    return processed_results


def parse_arguments():
    """
    Parse command line arguments with validation.

    Returns:
        argparse.Namespace: Validated command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Optimized GPU-First ONNX Inference for Images and Videos using IO Binding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--onnx",
        "-o",
        dest="onnx_model_path",
        type=Path,
        required=True,
        help="Path to the ONNX model file.",
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to the input image or video file.",
    )

    parser.add_argument(
        "--output-dir-d",
        dest="output_dir",
        type=Path,
        default="./outputs/",
        help="Output directory for the result. The filename is derived from the input.",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help=f"Batch size for video processing. Range: {CONFIG_RANGES['batch_size'][0]}-{CONFIG_RANGES['batch_size'][1]}.",
    )

    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.65,
        help=f"Confidence threshold for detections. Range: {CONFIG_RANGES['conf_threshold'][0]}-{CONFIG_RANGES['conf_threshold'][1]}.",
    )

    parser.add_argument(
        "--model-size",
        "-s",
        type=int,
        default=DEFAULT_MODEL_INPUT_SIZE,
        help=f"Input size (height/width) expected by the model. Range: {CONFIG_RANGES['model_size'][0]}-{CONFIG_RANGES['model_size'][1]}.",
    )

    parser.add_argument(
        "--cuda-device",
        "-g",
        type=int,
        default=0,
        help=f"CUDA device ID to use. Range: {CONFIG_RANGES['cuda_device_id'][0]}-{CONFIG_RANGES['cuda_device_id'][1]}.",
    )

    parser.add_argument(
        "--labels",
        "-l",
        dest="labels_dict",
        type=str,
        required=False,  # Changed from True to False
        default="{}",
        help='Optional dict of class labels in JSON format (e.g., \'{"0": "car", "1": "bike"}\'). If not provided, class IDs will be used directly.',
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="Prints debug info",
    )

    args = parser.parse_args()

    # Validate arguments
    try:
        # Parse the JSON labels string into a dict
        args.labels_dict = json.loads(args.labels_dict)

        # Resolve paths (expand user, make absolute)
        args.input_path = args.input_path.expanduser().resolve()
        args.onnx_model_path = args.onnx_model_path.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()

        # Check if input file exists
        if not args.input_path.exists():
            parser.error(f"Input file does not exist: {args.input_path}")

        # Check if model file exists
        if not args.onnx_model_path.exists():
            parser.error(f"ONNX model file does not exist: {args.onnx_model_path}")

        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate numeric parameters against defined ranges
        if not (
            CONFIG_RANGES["batch_size"][0]
            <= args.batch_size
            <= CONFIG_RANGES["batch_size"][1]
        ):
            parser.error(
                f"Batch size must be between {CONFIG_RANGES['batch_size'][0]} and {CONFIG_RANGES['batch_size'][1]}"
            )

        if not (
            CONFIG_RANGES["conf_threshold"][0]
            <= args.conf
            <= CONFIG_RANGES["conf_threshold"][1]
        ):
            parser.error(
                f"Confidence threshold must be between {CONFIG_RANGES['conf_threshold'][0]} and {CONFIG_RANGES['conf_threshold'][1]}"
            )

        if not (
            CONFIG_RANGES["model_size"][0]
            <= args.model_size
            <= CONFIG_RANGES["model_size"][1]
        ):
            parser.error(
                f"Model size must be between {CONFIG_RANGES['model_size'][0]} and {CONFIG_RANGES['model_size'][1]}"
            )

        if not (
            CONFIG_RANGES["cuda_device_id"][0]
            <= args.cuda_device
            <= CONFIG_RANGES["cuda_device_id"][1]
        ):
            parser.error(
                f"CUDA device ID must be between {CONFIG_RANGES['cuda_device_id'][0]} and {CONFIG_RANGES['cuda_device_id'][1]}"
            )

    except json.JSONDecodeError:
        parser.error("Invalid JSON format for labels.")
    except Exception as e:
        parser.error(str(e))

    return args


def main(
    input_path: Path,
    onnx_path: Path,
    labels_dict: dict,
    output_dir: Path = Path("outputs/"),
    batch_size: int = 32,
    conf_threshold: float = 0.65,
    model_input_size: int = DEFAULT_MODEL_INPUT_SIZE,
    cuda_device_id: int = 0,
    debug: bool = False,
):
    """
    Main function to orchestrate loading and processing.
    Expects fully resolved Path objects and validated parameters.

    Args:
        onnx_path: Path to the ONNX model file (resolved)
        input_path: Path to the input image or video file (resolved)
        labels_dict: Dict of class labels {0: "dog", 1: "cat"}
        output_dir: Output Dir
        batch_size: Batch size for video processing
        conf_threshold: Confidence threshold for detections
        model_input_size: Input size expected by the model
        cuda_device_id: CUDA device ID to use
        debug: Enable debug logging

    Returns:
        int: Exit code (0 for success)
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    onnx_path = Path(onnx_path)
    output_path = output_dir / f"{input_path.stem}_result{input_path.suffix}"
    labels_dict = {int(k): v for k, v in labels_dict.items()}  # ensure keys are ints

    try:
        logger.info(f"Processing {input_path} -> {output_path}")

        # Setup CUDA device
        if not setup_cuda_device(cuda_device_id):
            logger.error("Failed to setup CUDA device. Exiting.")
            return 1

        # Create configuration object
        config = Config(
            model_path=onnx_path,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            conf_threshold=conf_threshold,
            model_input_size=model_input_size,
            cuda_device_id=cuda_device_id,
            labels_dict=labels_dict,
        )

        # Initialize the Inferencer
        logger.debug(f"Initializing ONNX inference with model: {config.model_path}")
        inferencer = ONNXInferencer(config.model_path, device_id=config.cuda_device_id)

        # Print model information in debug mode
        if debug:
            logger.debug("Model Inputs:")
            for i, input_info in enumerate(inferencer.input_details):
                logger.debug(
                    f"  {i}: Name='{getattr(input_info, 'name', 'N/A')}', Shape={getattr(input_info, 'shape', 'N/A')}, Type={getattr(input_info, 'type', 'N/A')}"
                )
            logger.debug("Model Outputs:")
            for i, output_info in enumerate(inferencer.output_details):
                np_dtype = get_numpy_dtype_from_ort_type(
                    getattr(output_info, "type", "N/A")
                )
                logger.debug(
                    f"  {i}: Name='{getattr(output_info, 'name', 'N/A')}', Shape={getattr(output_info, 'shape', 'N/A')}, Type={getattr(output_info, 'type', 'N/A')} (NumPy: {np_dtype})"
                )

        # Determine input type and process
        is_image = is_image_file(config.input_path)

        if is_image:
            logger.debug(f"Processing single image: {config.input_path}")
            image_processor = ImageProcessor(
                inferencer=inferencer,
                class_names=config.labels_dict,
            )
            success = image_processor.process(
                img_path=config.input_path,
                output_path=config.output_path,
                conf_threshold=config.conf_threshold,
                model_input_size=config.model_input_size,
            )
            logger.debug("Image processing completed.")
            return 0 if success else 1
        elif config.input_path.is_dir():
            logger.debug(f"Processing images in directory: {config.input_path}")
            image_processor = ImageProcessor(
                inferencer=inferencer,
                class_names=config.labels_dict,
            )
            image_files = sorted([
                f for f in config.input_path.iterdir() if is_image_file(f)
            ])

            if not image_files:
                logger.warning(f"No image files found in: {config.input_path}")
                return 0

            all_batches_successful = True
            for i in range(0, len(image_files), config.batch_size):
                batch_paths = image_files[i : i + config.batch_size]
                batch_output_dir = output_dir
                batch_successful = image_processor.process_batch(
                    img_paths=batch_paths,
                    output_dir=batch_output_dir,
                    conf_threshold=config.conf_threshold,
                    model_input_size=config.model_input_size,
                )
                if not batch_successful:
                    all_batches_successful = False

            logger.debug("Batch image processing completed.")
            return 0 if all_batches_successful else 1
        else:
            logger.debug(f"Processing video: {config.input_path}")
            video_processor = VideoProcessor(
                inferencer=inferencer,
                video_path=config.input_path,
                output_path=config.output_path,
                class_names=config.labels_dict,
                batch_size=config.batch_size,
                conf_threshold=config.conf_threshold,
                model_input_size=config.model_input_size,
            )
            video_processor.run()
            logger.debug("Video processing completed.")
            return 0

    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"Configuration or File error: {e}")
        logger.error(traceback.format_exc())
        return 1
    except (RuntimeError, IOError) as e:
        logger.error(f"Runtime error during processing: {e}")
        logger.error(traceback.format_exc())
        return 1
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Final cleanup
        logger.debug("Performing final cleanup.")
        # Force CUDA memory cleanup at the end
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {e}")
            logger.error(traceback.format_exc())
            pass  # Avoid crashing during cleanup


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Convert labels string to dictionary if it's a string
    if isinstance(args.labels_dict, str):
        try:
            labels_dict = json.loads(args.labels_dict)
        except json.JSONDecodeError:
            logger.error("Failed to parse labels JSON. Using empty dictionary.")
            labels_dict = {}
    else:
        labels_dict = args.labels_dict

    # Determine output path
    if args.input_path.is_dir():
        output_path = args.output_dir
    else:
        output_path = (
            args.output_dir / f"{args.input_path.stem}_result{args.input_path.suffix}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Call the main function with the parsed arguments
    exit_code = main(
        input_path=args.input_path,
        onnx_path=args.onnx_model_path,
        labels_dict=labels_dict,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        conf_threshold=args.conf,
        model_input_size=args.model_size,
        cuda_device_id=args.cuda_device,
        debug=args.debug,
    )

    # Exit with the code returned by main
    sys.exit(exit_code)
