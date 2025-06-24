"""
Annotation package for object detection visualization.
Provides utilities for annotating frames with detection results.
Supports batch processing for improved performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from supervision.annotators.core import BoxAnnotator, RichLabelAnnotator
    from supervision.geometry.core import Position
    from supervision.detection.core import Detections
except ImportError:
    raise ImportError(
        "Supervision package is required. Install it with 'pip install supervision'"
    )


@dataclass(frozen=True)
class ResizeInfo:
    """Stores information about the resizing and padding process."""
    ratio: float
    pad_w: int
    pad_h: int
    original_height: int
    original_width: int
    
    @classmethod
    def batch_from_arrays(cls, 
                         ratios: List[float], 
                         pad_ws: List[int], 
                         pad_hs: List[int], 
                         orig_heights: List[int], 
                         orig_widths: List[int]) -> List["ResizeInfo"]:
        """Create a batch of ResizeInfo objects from parallel arrays."""
        batch_size = len(ratios)
        return [
            cls(
                ratio=ratios[i],
                pad_w=pad_ws[i],
                pad_h=pad_hs[i],
                original_height=orig_heights[i],
                original_width=orig_widths[i]
            )
            for i in range(batch_size)
        ]


@dataclass(frozen=True)
class DetectionResult:
    """Stores a single detection result after post-processing."""
    box: np.ndarray  # Shape (4,) representing xyxy
    score: float
    label: int


def annotate_frames(
    frames: List[np.ndarray],  # Original frames (RGB format)
    processed_detections: List[List[DetectionResult]],  # Detections per frame
    class_names: Dict[int, str],
    thickness: int = 2,
    font_size: int = 12
) -> List[np.ndarray]:
    """
    Annotate frames with processed detections using Supervision.

    Args:
        frames: List of original frames (RGB format)
        processed_detections: List of detection results per frame
        class_names: Dictionary mapping class IDs to names for labels
        thickness: Line thickness for boxes
        font_size: Font size for labels

    Returns:
        List[np.ndarray]: List of annotated frames
    """
    # Create annotator (reused for all frames)
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
    )
    annotated_frames = []

    for i, (frame, detections) in enumerate(zip(frames, processed_detections)):
        # If no detections for this frame, return a copy of the original
        if not detections:
            annotated_frames.append(frame.copy())
            continue

        # Prepare data for supervision.Detections from DetectionResult dataclass
        boxes_np = np.array([det.box for det in detections])
        scores_np = np.array([det.score for det in detections])
        labels_np = np.array([det.label for det in detections]).astype(int)

        # Create Supervision Detections object
        sv_detections = Detections(
            xyxy=boxes_np,
            confidence=scores_np,
            class_id=labels_np,
        )

        # Create labels for annotation (e.g., "class: score")
        labels = []
        for class_id, confidence in zip(
            sv_detections.class_id, sv_detections.confidence # type: ignore
        ):
            label = class_names.get(int(class_id), "unknown")
            labels.append(f"{label}: {confidence:.2f}")

        # Annotate the frame (operates on a copy)
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=sv_detections,
        )
        annotated_frame = text_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            labels=labels,
        )

        annotated_frames.append(annotated_frame)

    return annotated_frames


def annotate_detections(
    image: np.ndarray,  # RGB format image
    boxes: np.ndarray,  # xyxy format boxes 
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    conf_threshold: float = 0.5,
    thickness: int = 2,
    font_size: int = 12
) -> np.ndarray:
    """
    Annotate a single image with detection results.

    Args:
        image: Input image (RGB format)
        boxes: Bounding boxes in xyxy format
        scores: Confidence scores
        labels: Class labels
        class_names: Dictionary mapping class IDs to names for labels
        conf_threshold: Confidence threshold for filtering detections
        thickness: Line thickness for boxes
        font_size: Font size for labels

    Returns:
        np.ndarray: Annotated image
    """
    # Filter by confidence
    mask = scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    
    # Create annotators
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
    )
    
    # Create Supervision Detections object
    sv_detections = Detections(
        xyxy=filtered_boxes,
        confidence=filtered_scores,
        class_id=filtered_labels,
    )
    
    # Create labels for annotation
    annotation_labels = []
    for class_id, confidence in zip(
        sv_detections.class_id, sv_detections.confidence # type: ignore
    ):
        label = class_names.get(int(class_id), "unknown")
        annotation_labels.append(f"{label}: {confidence:.2f}")
    
    # Annotate the image
    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=sv_detections,
    )
    annotated_image = text_annotator.annotate(
        scene=annotated_image,
        detections=sv_detections,
        labels=annotation_labels,
    )
    
    return annotated_image


def annotate_batch(
    images: List[np.ndarray],  # RGB format images
    boxes_batch: List[np.ndarray],  # xyxy format boxes for each image
    scores_batch: List[np.ndarray],  # scores for each image
    labels_batch: List[np.ndarray],  # labels for each image
    class_names: Dict[int, str],
    conf_threshold: float = 0.5,
    thickness: int = 2,
    font_size: int = 12
) -> List[np.ndarray]:
    """
    Annotate a batch of images with detection results.

    Args:
        images: List of input images (RGB format)
        boxes_batch: List of bounding boxes arrays (one per image)
        scores_batch: List of score arrays (one per image)
        labels_batch: List of label arrays (one per image)
        class_names: Dictionary mapping class IDs to names for labels
        conf_threshold: Confidence threshold for filtering detections
        thickness: Line thickness for boxes
        font_size: Font size for labels

    Returns:
        List[np.ndarray]: List of annotated images
    """
    # Create annotators (reused for all images)
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
    )
    
    annotated_images = []
    
    for i, (image, boxes, scores, labels) in enumerate(zip(images, boxes_batch, scores_batch, labels_batch)):
        # Filter by confidence
        mask = scores > conf_threshold
        if not np.any(mask):
            # No detections above threshold
            annotated_images.append(image.copy())
            continue
            
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        
        # Create Supervision Detections object
        sv_detections = Detections(
            xyxy=filtered_boxes,
            confidence=filtered_scores,
            class_id=filtered_labels,
        )
        
        # Create labels for annotation
        annotation_labels = []
        for class_id, confidence in zip(
            sv_detections.class_id, sv_detections.confidence # type: ignore
        ):
            label = class_names.get(int(class_id), "unknown")
            annotation_labels.append(f"{label}: {confidence:.2f}")
        
        # Annotate the image
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=sv_detections,
        )
        annotated_image = text_annotator.annotate(
            scene=annotated_image,
            detections=sv_detections,
            labels=annotation_labels,
        )
        
        annotated_images.append(annotated_image)
    
    return annotated_images


def apply_detection_preprocessing(
    boxes: np.ndarray,  # xyxy format
    ratio: float,
    pad_w: int,
    pad_h: int,
    original_width: int,
    original_height: int
) -> np.ndarray:
    """
    Apply preprocessing to detection boxes to account for resizing and padding.
    Maps boxes from model output space back to original image space.

    Args:
        boxes: Boxes in xyxy format from model
        ratio: Resize ratio used during preprocessing
        pad_w: Padding width
        pad_h: Padding height
        original_width: Original image width
        original_height: Original image height

    Returns:
        np.ndarray: Boxes adjusted to original image coordinates
    """
    adjusted_boxes = boxes.copy()
    
    # Adjust bounding boxes according to the resizing and padding
    adjusted_boxes[:, 0] = (adjusted_boxes[:, 0] - pad_w) / ratio
    adjusted_boxes[:, 1] = (adjusted_boxes[:, 1] - pad_h) / ratio
    adjusted_boxes[:, 2] = (adjusted_boxes[:, 2] - pad_w) / ratio
    adjusted_boxes[:, 3] = (adjusted_boxes[:, 3] - pad_h) / ratio
    
    # Clip to image boundaries
    adjusted_boxes[:, 0] = np.clip(adjusted_boxes[:, 0], 0, original_width)
    adjusted_boxes[:, 1] = np.clip(adjusted_boxes[:, 1], 0, original_height)
    adjusted_boxes[:, 2] = np.clip(adjusted_boxes[:, 2], 0, original_width)
    adjusted_boxes[:, 3] = np.clip(adjusted_boxes[:, 3], 0, original_height)
    
    return adjusted_boxes


def apply_detection_preprocessing_batch(
    boxes_batch: List[np.ndarray],  # List of boxes arrays (one per image)
    resize_infos: List[ResizeInfo]  # Resize info for each image
) -> List[np.ndarray]:
    """
    Apply preprocessing to a batch of detection boxes.
    
    Args:
        boxes_batch: List of boxes arrays (one per image)
        resize_infos: List of resize info objects (one per image)
        
    Returns:
        List[np.ndarray]: List of adjusted boxes arrays
    """
    adjusted_boxes_batch = []
    
    for boxes, resize_info in zip(boxes_batch, resize_infos):
        adjusted_boxes = apply_detection_preprocessing(
            boxes=boxes,
            ratio=resize_info.ratio,
            pad_w=resize_info.pad_w,
            pad_h=resize_info.pad_h,
            original_width=resize_info.original_width,
            original_height=resize_info.original_height
        )
        adjusted_boxes_batch.append(adjusted_boxes)
    
    return adjusted_boxes_batch


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, ResizeInfo]:
    """
    Resizes an image while maintaining aspect ratio and pads it.

    Args:
        image: Input image (NumPy array, HWC format)
        target_size: Target size for resizing

    Returns:
        Tuple[np.ndarray, ResizeInfo]: Resized image and resize information
    """
    original_height, original_width = image.shape[:2]
    
    # Calculate new dimensions while preserving aspect ratio
    ratio = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Calculate padding
    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2
    
    # Create padded image
    padded_image = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    padded_image[pad_h:pad_h+new_height, pad_w:pad_w+new_width] = resized_image
    
    resize_info = ResizeInfo(
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
        original_height=original_height,
        original_width=original_width,
    )
    
    return padded_image, resize_info


def resize_batch_with_aspect_ratio(
    images: List[np.ndarray],
    target_size: int
) -> Tuple[List[np.ndarray], List[ResizeInfo]]:
    """
    Resizes a batch of images while maintaining aspect ratio and pads them.
    
    Args:
        images: List of input images (NumPy arrays, HWC format)
        target_size: Target size for resizing
        
    Returns:
        Tuple[List[np.ndarray], List[ResizeInfo]]: List of resized images and their resize information
    """
    resized_images = []
    resize_infos = []
    
    for image in images:
        resized_image, resize_info = resize_with_aspect_ratio(image, target_size)
        resized_images.append(resized_image)
        resize_infos.append(resize_info)
    
    return resized_images, resize_infos