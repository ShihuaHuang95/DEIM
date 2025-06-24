"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import json
import os
from typing import List, Dict
from glob import glob

# Import from our annotation package
from annotate import (
    annotate_detections, 
    apply_detection_preprocessing,
    resize_with_aspect_ratio,
    annotate_batch,
    apply_detection_preprocessing_batch,
    resize_batch_with_aspect_ratio
)


def process_image_batch(sess, image_paths: List[str], output_dir: str, class_names: Dict[int, str], batch_size: int = 4):
    """
    Process a batch of images with the ONNX model.
    
    Args:
        sess: ONNX Runtime inference session
        image_paths: List of paths to images to process
        output_dir: Directory to save processed images
        class_names: Dictionary mapping class IDs to names
        batch_size: Number of images to process at once
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}, images {i+1}-{min(i+batch_size, len(image_paths))}")
        
        # Load images
        images_pil = [Image.open(path).convert('RGB') for path in batch_paths]
        images_np = [np.array(img) for img in images_pil]
        
        # Resize images
        resized_images, resize_infos = resize_batch_with_aspect_ratio(images_np, 640)
        
        # Convert to tensors
        transforms = T.Compose([T.ToTensor()])
        batch_tensors = [transforms(Image.fromarray(img)).unsqueeze(0) for img in resized_images] # type: ignore
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # Create orig_target_sizes tensor
        orig_sizes_tensor = torch.tensor([[img.shape[0], img.shape[1]] for img in resized_images])
        
        # Run inference
        output = sess.run(
            output_names=None,
            input_feed={'images': batch_tensor.numpy(), "orig_target_sizes": orig_sizes_tensor.numpy()}
        )
        
        labels, boxes, scores = output
        
        # Process boxes for each image
        boxes_batch = [boxes[j] for j in range(len(batch_paths))]
        
        # Process boxes to original image coordinates using the batch function
        adjusted_boxes_batch = apply_detection_preprocessing_batch(
            boxes_batch=boxes_batch,
            resize_infos=resize_infos
        )
        
        # Annotate images
        annotated_images = annotate_batch(
            images=images_np,
            boxes_batch=adjusted_boxes_batch,
            scores_batch=[scores[j] for j in range(len(batch_paths))],
            labels_batch=[labels[j] for j in range(len(batch_paths))],
            class_names=class_names,
            conf_threshold=0.4
        )
        
        # Save results
        for j, annotated_img in enumerate(annotated_images):
            output_path = os.path.join(output_dir, os.path.basename(batch_paths[j]))
            Image.fromarray(annotated_img).save(output_path)
        
    print(f"Batch image processing complete. Results saved in '{output_dir}'.")


def process_image(sess, im_pil, output_path: str, class_names: Dict[int, str]):
    """Process a single image with the ONNX model."""
    # Convert PIL image to numpy array for our package's resize function
    im_np = np.array(im_pil)
    
    # Resize image while preserving aspect ratio using our package
    resized_im_np, resize_info = resize_with_aspect_ratio(im_np, 640)
    ratio = resize_info.ratio
    pad_w = resize_info.pad_w
    pad_h = resize_info.pad_h
    
    # Convert back to PIL for the model
    resized_im_pil = Image.fromarray(resized_im_np)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
        T.ToTensor(),
    ])
    im_data: torch.Tensor = transforms(resized_im_pil).unsqueeze(0) # type: ignore

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )

    labels, boxes, scores = output
    
    # Process boxes for the first image in batch
    adjusted_boxes = apply_detection_preprocessing(
        boxes=boxes[0],
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
        original_width=im_pil.size[0],
        original_height=im_pil.size[1]
    )
    
    # Annotate the image
    result_image = annotate_detections(
        image=im_np,
        boxes=adjusted_boxes,
        scores=scores[0],
        labels=labels[0],
        class_names=class_names,
        conf_threshold=0.4
    )
    
    # Convert back to PIL and save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Image.fromarray(result_image).save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video(sess, video_path, output_path: str, class_names: Dict[int, str], batch_frames: int = 8):
    """Process a video with the ONNX model, using batch processing for frames."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video with {total_frames} frames...")
    
    # Process frames in batches
    while cap.isOpened():
        # Read batch of frames
        frames = []
        for _ in range(batch_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        if not frames:
            break
        
        # Resize frames
        resized_frames, resize_infos = resize_batch_with_aspect_ratio(frames, 640)
        
        # Convert to tensors
        transforms = T.Compose([T.ToTensor()])
        batch_tensors = [transforms(Image.fromarray(img)).unsqueeze(0) for img in resized_frames] # type: ignore
        
        # If we have less than batch_frames, we still need to handle it properly
        if len(batch_tensors) < batch_frames:
            batch_tensor = torch.cat(batch_tensors, dim=0)
            orig_sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in resized_frames])
        else:
            batch_tensor = torch.cat(batch_tensors, dim=0)
            orig_sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in resized_frames])
        
        # Run inference
        output = sess.run(
            output_names=None,
            input_feed={'images': batch_tensor.numpy(), "orig_target_sizes": orig_sizes.numpy()}
        )
        
        labels, boxes, scores = output
        
        # Process boxes for each frame
        boxes_batch = [boxes[i] for i in range(len(frames))]
        
        # Process boxes to original image coordinates using the batch function
        adjusted_boxes_batch = apply_detection_preprocessing_batch(
            boxes_batch=boxes_batch,
            resize_infos=resize_infos
        )
        
        # Annotate frames
        annotated_frames = annotate_batch(
            images=frames,
            boxes_batch=adjusted_boxes_batch,
            scores_batch=[scores[i] for i in range(len(frames))],
            labels_batch=[labels[i] for i in range(len(frames))],
            class_names=class_names,
            conf_threshold=0.4
        )
        
        # Write frames to video
        for annotated_frame in annotated_frames:
            # Convert RGB to BGR for OpenCV
            frame_with_detections = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_with_detections)
            frame_count += 1
        
        print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{output_path}'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.onnx,
        sess_options=sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print(f"Using device: {ort.get_device()}")
    
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
            output_path = "onnx_results"
        else:
            output_path = "onnx_result.jpg" if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) else "onnx_result.mp4"

    # Check if input is a directory (batch mode)
    if os.path.isdir(args.input):
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.input, f"*{ext}")))
            image_paths.extend(glob(os.path.join(args.input, f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        process_image_batch(sess, image_paths, output_path, class_names, args.batch_size)
    else:
        # Single file processing
        input_path = args.input
        try:
            # Try to open the input as an image
            im_pil = Image.open(input_path).convert('RGB')
            process_image(sess, im_pil, output_path, class_names)
        except IOError:
            # Not an image, process as video
            process_video(sess, input_path, output_path, class_names, args.batch_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image, video file, or directory of images.')
    parser.add_argument('--output', type=str, help='Path to save the output. For batch processing, this is a directory.')
    parser.add_argument('--labels', type=str, help='JSON string mapping class IDs to names. Example: \'{"0": "person", "1": "car"}\'')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing multiple images or video frames.')
    args = parser.parse_args()
    main(args)