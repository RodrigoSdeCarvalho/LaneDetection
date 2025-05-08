from essentials.add_module import set_working_directory
set_working_directory()

import os
import cv2
import time
import json
import numpy as np
import torch
from typing import Dict

from neural_networks.decorators.lane_detector.lane_detector import LaneDetector
from api import API
from camera.dummy_camera import DummyCamera
from utils.path import Path
from utils.logger import Logger
from app.metrics import LaneMetrics

class TestInferenceViewer:
    """
    Displays the inference results of the lane detection model on the test dataset
    as a video-like visualization, showing each image for 5 seconds.
    """
    
    def __init__(self, test_labels_path: str = None):
        """
        Initialize the test inference viewer.
        
        Args:
            test_labels_path: Path to the test labels JSON file.
                If None, uses the default test_label_fixed.json file.
        """
        self.metrics = LaneMetrics(test_labels_path=test_labels_path)
        self.camera = DummyCamera()
        self.api = API(self.camera)
        self.display_time = 5  # Time in seconds to display each image
        
    def run(self, num_samples: int = None, shuffle: bool = False):
        """
        Run the inference viewer on test images.
        
        Args:
            num_samples: Number of samples to process (None for all)
            shuffle: Whether to shuffle the test data before displaying
        """
        # Get test data from the metrics object
        test_data = self.metrics.test_data.copy()
        
        if shuffle:
            import random
            random.shuffle(test_data)
        
        if num_samples is not None:
            samples = test_data[:num_samples]
        else:
            samples = test_data
            
        Logger.info(f"Running inference viewer on {len(samples)} images...")
        
        # Create a window for display
        window_name = "Lane Detection Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Process each sample
        for i, sample in enumerate(samples):
            image_path = sample["raw_file"]
            
            # Get the actual file path
            actual_path = self.metrics._find_image_file(image_path)
            
            # Load the image
            image = cv2.imread(actual_path)
            if image is None:
                Logger.error(f"Failed to load image: {actual_path}")
                continue
                
            # Create visualization with ground truth and prediction
            vis_image = self.create_visualization(image, sample)
            
            # Display image info
            progress_text = f"Image {i+1}/{len(samples)}"
            file_text = f"File: {os.path.basename(image_path)}"
            
            # Add text to visualization
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            # Add background rectangle for better text visibility
            (text_width1, text_height1), _ = cv2.getTextSize(progress_text, font, font_scale, font_thickness)
            (text_width2, text_height2), _ = cv2.getTextSize(file_text, font, font_scale, font_thickness)
            max_width = max(text_width1, text_width2)
            
            cv2.rectangle(vis_image, (10, 10), (20 + max_width, 20 + text_height1 + text_height2 + 10), bg_color, -1)
            cv2.putText(vis_image, progress_text, (15, 15 + text_height1), font, font_scale, text_color, font_thickness)
            cv2.putText(vis_image, file_text, (15, 15 + text_height1 + text_height2 + 5), font, font_scale, text_color, font_thickness)
            
            # Display the image
            cv2.imshow(window_name, vis_image)
            
            # Wait for the specified display time or until key press
            key = cv2.waitKey(1)
            start_time = time.time()
            while (time.time() - start_time) < self.display_time:
                key = cv2.waitKey(100)  # Check for key press every 100ms
                if key == 27:  # ESC key
                    break
            
            # Exit if ESC key is pressed
            if key == 27:
                break
                
        # Clean up
        cv2.destroyAllWindows()
        Logger.info("Inference viewer completed.")
        
    def create_visualization(self, image: np.ndarray, gt_data: Dict) -> np.ndarray:
        """
        Create a visualization of the model's inference alongside ground truth.
        
        Args:
            image: Input image
            gt_data: Ground truth data for the image
            
        Returns:
            Visualization image with predictions and ground truth
        """
        # Resize image to model's default size
        resized_image = cv2.resize(image, LaneDetector.DEFAULT_IMAGE_SIZE)
        
        # Run the lane detector
        instances_map = self.api._detector(image)
        
        # Create visualization image
        vis_image = resized_image.copy()
        
        # Create predicted lanes overlay
        pred_overlay = np.zeros_like(resized_image)
        if instances_map is not None:
            # Convert to binary mask and extract individual lanes
            pred_lanes = self.metrics._extract_lane_masks_from_instances(instances_map)
            
            # Assign different colors to each lane
            colors = [
                (255, 0, 0),   # Blue
                (0, 255, 0),   # Green
                (0, 0, 255),   # Red
                (0, 255, 255), # Yellow
                (255, 0, 255), # Magenta
                (255, 255, 0)  # Cyan
            ]
            
            for i, lane_mask in enumerate(pred_lanes):
                color = colors[i % len(colors)]
                pred_overlay[lane_mask > 0] = color
        
        # Create ground truth lanes overlay (white)
        gt_overlay = np.zeros_like(resized_image)
        gt_binary = self.metrics._convert_tusimple_to_mask(gt_data["lanes"], gt_data["h_samples"])
        gt_overlay[gt_binary > 0] = (255, 255, 255)
        
        # Blend overlays with original image
        alpha_pred = 0.6  # Transparency for predictions
        alpha_gt = 0.4    # Transparency for ground truth
        
        # Combine image with prediction and ground truth overlays
        vis_image = cv2.addWeighted(vis_image, 1.0, pred_overlay, alpha_pred, 0)
        vis_image = cv2.addWeighted(vis_image, 1.0, gt_overlay, alpha_gt, 0)
        
        # Calculate metrics for this image
        if instances_map is not None:
            pred_binary = (instances_map > 0).cpu().numpy().astype(np.uint8)
            iou = self.metrics.calculate_iou(pred_binary, gt_binary)
            
            pred_lanes = self.metrics._extract_lane_masks_from_instances(instances_map)
            gt_lanes = self.metrics._extract_individual_gt_lanes(gt_data["lanes"], gt_data["h_samples"])
            conf_matrix = self.metrics.calculate_per_lane_confusion_matrix(pred_lanes, gt_lanes)
            
            # Add metrics as text
            iou_text = f"IoU: {iou:.4f}"
            tp_text = f"TP: {conf_matrix['tp']}"
            fp_text = f"FP: {conf_matrix['fp']}"
            fn_text = f"FN: {conf_matrix['fn']}"
            
            # Get center of lane
            center_point = self.api._detector.get_center_of_lane(instances_map)
            if center_point is not None:
                # Draw a prominent circle at the center point
                cv2.circle(vis_image, (int(center_point[1]), int(center_point[0])), 10, (0, 255, 255), -1)  # Yellow filled circle
                cv2.circle(vis_image, (int(center_point[1]), int(center_point[0])), 12, (0, 0, 0), 2)       # Black outline
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            # Add background rectangle for better text visibility
            for i, text in enumerate([iou_text, tp_text, fp_text, fn_text]):
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(vis_image, (10, 80 + i*25 - 15), (10 + text_width, 80 + i*25 + 5), bg_color, -1)
                cv2.putText(vis_image, text, (10, 80 + i*25), font, font_scale, text_color, font_thickness)
        
        # Add legend
        legend_texts = ["Predicted Lanes", "Ground Truth", "Lane Center"]
        legend_colors = [(255, 0, 0), (255, 255, 255), (0, 255, 255)]  # Blue for predicted lanes, White for ground truth, Yellow for center
        
        for i, (text, color) in enumerate(zip(legend_texts, legend_colors)):
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(vis_image, (vis_image.shape[1] - 10 - text_width - 30, 20 + i*25 - 15), 
                         (vis_image.shape[1] - 10, 20 + i*25 + 5), bg_color, -1)
            cv2.putText(vis_image, text, (vis_image.shape[1] - 10 - text_width - 20, 20 + i*25), 
                       font, font_scale, text_color, font_thickness)
            
            # Draw circle for center point legend
            if i == 2:  # Lane Center legend
                center_x = vis_image.shape[1] - 10 - 7
                center_y = 20 + i*25 - 5
                cv2.circle(vis_image, (center_x, center_y), 5, color, -1)
                cv2.circle(vis_image, (center_x, center_y), 6, (0, 0, 0), 1)
            else:  # Regular colored rectangle for other legend items
                cv2.rectangle(vis_image, (vis_image.shape[1] - 10 - 15, 20 + i*25 - 10), 
                             (vis_image.shape[1] - 10, 20 + i*25), color, -1)
        
        return vis_image

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Display lane detection inference on test images")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to display")
    parser.add_argument("--test_labels", type=str, default=None, help="Path to test labels JSON file")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the test data before displaying")
    parser.add_argument("--display_time", type=int, default=5, help="Time in seconds to display each image")

    args = parser.parse_args()

    viewer = TestInferenceViewer(test_labels_path=args.test_labels)
    viewer.display_time = args.display_time
    viewer.run(num_samples=args.num_samples, shuffle=args.shuffle)
