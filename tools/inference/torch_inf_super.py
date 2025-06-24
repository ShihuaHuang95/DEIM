"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import json
import os
import sys
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Import from our new annotation package
from annotate import (
    annotate_batch,
    annotate_detections,
    apply_detection_preprocessing,
    apply_detection_preprocessing_batch,
    resize_batch_with_aspect_ratio,
    resize_with_aspect_ratio,
)
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig


def process_image_batch(
    model,
    device,
    image_paths: List[str],
    output_dir: str,
    class_names: Dict[int, str],
    batch_size: int = 4,
    conf_threshold: float = 0.4,
    model_img_size: int = 640
):
    """
    Process a batch of images with the PyTorch model.

    Args:
        model: PyTorch model
        device: Device to run inference on
        image_paths: List of paths to images to process
        output_dir: Directory to save processed images
        class_names: Dictionary mapping class IDs to names
        batch_size: Number of images to process at once
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}, images {i + 1}-{min(i + batch_size, len(image_paths))}"
        )

        # Load images
        images_pil = [Image.open(path).convert("RGB") for path in batch_paths]
        images_np = [np.array(img) for img in images_pil]

        # Resize images using our annotation package
        resized_images, resize_infos = resize_batch_with_aspect_ratio(images_np, model_img_size)

        # Convert to tensors
        transforms = T.Compose([T.ToTensor()])
        batch_tensors = [
            transforms(Image.fromarray(img)).unsqueeze(0)  # type: ignore
            for img in resized_images  # type: ignore
        ]  # type: ignore
        batch_tensor = torch.cat(batch_tensors, dim=0)

        # Create orig_target_sizes tensor (height, width format for PyTorch model)
        orig_sizes_tensor = torch.tensor([
            [img.shape[0], img.shape[1]] for img in resized_images
        ]).to(device)

        # Run model in eval mode
        with torch.no_grad():
            output = model(batch_tensor.to(device), orig_sizes_tensor)

        labels, boxes, scores = output

        # Convert tensors to numpy arrays for each item in batch
        boxes_batch = []
        scores_batch = []
        labels_batch = []

        for j in range(len(batch_paths)):
            if isinstance(boxes[j], torch.Tensor):
                boxes_batch.append(boxes[j].cpu().numpy())
                scores_batch.append(scores[j].cpu().numpy())
                labels_batch.append(labels[j].cpu().numpy())
            else:
                boxes_batch.append(boxes[j])
                scores_batch.append(scores[j])
                labels_batch.append(labels[j])

        # Process boxes to original image coordinates
        adjusted_boxes_batch = apply_detection_preprocessing_batch(
            boxes_batch=boxes_batch, resize_infos=resize_infos
        )

        # Annotate images
        annotated_images = annotate_batch(
            images=images_np,
            boxes_batch=adjusted_boxes_batch,
            scores_batch=scores_batch,
            labels_batch=labels_batch,
            class_names=class_names,
            conf_threshold=conf_threshold,
        )

        # Save results
        for j, annotated_img in enumerate(annotated_images):
            output_path = os.path.join(output_dir, os.path.basename(batch_paths[j]))
            Image.fromarray(annotated_img).save(output_path)

    print(f"Batch image processing complete. Results saved in '{output_dir}'.")


def process_image(
    model,
    device,
    file_path: str,
    output_path: str,
    class_names: Dict[int, str],
    conf_threshold: float = 0.4,
    model_img_size: int = 640
):
    """Process a single image with the PyTorch model."""
    im_pil = Image.open(file_path).convert("RGB")
    im_np = np.array(im_pil)

    # Resize image while preserving aspect ratio using our package
    resized_im_np, resize_info = resize_with_aspect_ratio(im_np, model_img_size)

    # Convert to tensor for the model
    transforms = T.Compose([T.ToTensor()])
    im_data = transforms(Image.fromarray(resized_im_np)).unsqueeze(0)  # type: ignore

    # Original size in h,w format for the model
    orig_size = torch.tensor([[resized_im_np.shape[0], resized_im_np.shape[1]]]).to(
        device
    )

    # Run model in eval mode
    with torch.no_grad():
        output = model(im_data.to(device), orig_size)

    labels, boxes, scores = output

    # Convert boxes and labels to numpy arrays if they're tensors
    if isinstance(boxes[0], torch.Tensor):
        boxes_np = boxes[0].cpu().numpy()
        scores_np = scores[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()
    else:
        boxes_np = boxes[0]
        scores_np = scores[0]
        labels_np = labels[0]

    # Process boxes to original image coordinates
    adjusted_boxes = apply_detection_preprocessing(
        boxes=boxes_np,
        ratio=resize_info.ratio,
        pad_w=resize_info.pad_w,
        pad_h=resize_info.pad_h,
        original_width=im_pil.size[0],
        original_height=im_pil.size[1],
    )

    # Annotate the image
    result_image = annotate_detections(
        image=im_np,
        boxes=adjusted_boxes,
        scores=scores_np,
        labels=labels_np,
        class_names=class_names,
        conf_threshold=conf_threshold,
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save the result
    Image.fromarray(result_image).save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video(
    model,
    device,
    file_path: str,
    output_path: str,
    class_names: Dict[int, str],
    batch_size: int = 8,
    conf_threshold: float = 0.75,
    model_img_size: int = 640
):
    """Process a video with the PyTorch model, using batch processing for frames."""
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([T.ToTensor()])

    frame_count = 0
    print(f"Processing video with {total_frames} frames...")

    while cap.isOpened():
        # Read batch of frames
        frames = []

        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if not frames:
            break

        # Resize frames using our annotation package
        resized_frames, resize_infos = resize_batch_with_aspect_ratio(frames, model_img_size)

        # Convert to tensors
        batch_tensors = [
            transforms(Image.fromarray(img)).unsqueeze(0)  # type: ignore
            for img in resized_frames  # type: ignore
        ]  # type: ignore
        batch_tensor = torch.cat(batch_tensors, dim=0)

        # Create orig_target_sizes tensor (height, width format for PyTorch model)
        orig_sizes_tensor = torch.tensor([
            [img.shape[0], img.shape[1]] for img in resized_frames
        ]).to(device)

        # Run inference in eval mode
        with torch.no_grad():
            output = model(batch_tensor.to(device), orig_sizes_tensor)

        labels, boxes, scores = output

        # Convert tensors to numpy arrays for each frame
        boxes_batch = []
        scores_batch = []
        labels_batch = []

        for i in range(len(frames)):
            if isinstance(boxes[i], torch.Tensor):
                boxes_batch.append(boxes[i].cpu().numpy())
                scores_batch.append(scores[i].cpu().numpy())
                labels_batch.append(labels[i].cpu().numpy())
            else:
                boxes_batch.append(boxes[i])
                scores_batch.append(scores[i])
                labels_batch.append(labels[i])

        # Process boxes to original image coordinates
        adjusted_boxes_batch = apply_detection_preprocessing_batch(
            boxes_batch=boxes_batch, resize_infos=resize_infos
        )

        # Annotate frames
        annotated_frames = annotate_batch(
            images=frames,
            boxes_batch=adjusted_boxes_batch,
            scores_batch=scores_batch,
            labels_batch=labels_batch,
            class_names=class_names,
            conf_threshold=conf_threshold,
        )

        # Write frames to video
        for annotated_frame in annotated_frames:
            # Convert RGB to BGR for OpenCV
            frame_with_detections = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_with_detections)
            frame_count += 1


        print(
            f"{batch_size}Processed {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)..."
        )

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{output_path}'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()  # type: ignore
            self.postprocessor = cfg.postprocessor.deploy()  # type: ignore

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    model.eval()  # Set model to evaluation mode

    # Parse class names from labels argument if provided
    if args.labels:
        try:
            class_names = json.loads(args.labels)
            # Convert keys to integers
            class_names = {int(k): v for k, v in class_names.items()}
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for labels. Using default labels.")
            class_names = {i: f"class_{i}" for i in range(100)}
    else:
        # Default class names
        class_names = {i: f"class_{i}" for i in range(100)}

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default output path
        if os.path.isdir(args.input):
            output_path = "torch_results"
        else:
            output_path = (
                "torch_results.jpg"
                if args.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                else "torch_results.mp4"
            )

    # Check if the input is a directory (batch mode)
    if os.path.isdir(args.input):
        # Get all image files in the directory
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.input, f"*{ext}")))
            image_paths.extend(glob(os.path.join(args.input, f"*{ext.upper()}")))

        if not image_paths:
            print(f"No images found in {args.input}")
            return

        print(f"Found {len(image_paths)} images to process")
        process_image_batch(
            model,
            device,
            image_paths,
            output_path,
            class_names,
            args.batch_size,
            args.conf_threshold,
            args.model_img_size
        )
    else:
        # Check if the input file is an image or a video
        file_path = args.input
        if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Process as image
            process_image(
                model, device, file_path, output_path, class_names, args.conf_threshold, args.model_img_size
            )
        else:
            # Process as video
            process_video(
                model,
                device,
                file_path,
                output_path,
                class_names,
                args.batch_size,
                args.conf_threshold,
                args.model_img_size
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the output. For batch processing, this is a directory.",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        help='JSON string mapping class IDs to names. Example: \'{"0": "person", "1": "car"}\'',
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing multiple images or video frames.",
    )

    parser.add_argument(
        "-t",
        "--conf-threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for filtering detections.",
    )
    
    parser.add_argument("-s", "--model_img_size", type=int, default=640)

    args = parser.parse_args()
    main(args)
