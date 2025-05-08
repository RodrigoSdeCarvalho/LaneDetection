from essentials.add_module import set_working_directory
set_working_directory()

import json
import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Any
from scipy.optimize import linear_sum_assignment

from neural_networks.decorators.lane_detector.lane_detector import LaneDetector
from api import API
from camera.dummy_camera import DummyCamera
from utils.path import Path
from utils.logger import Logger


class LaneMetrics:
    """
    Class for evaluating lane detection metrics against TuSimple ground truth.
    Implements:
    1. IoU (Intersection over Union) for lane segmentation
    2. Per-lane confusion matrix (TP, FP, FN)
    """
    
    def __init__(self, test_labels_path: str = None):
        """
        Initialize the metrics calculator.
        
        Args:
            test_labels_path: Path to the test labels JSON file.
                If None, uses the default test_label_fixed.json file.
        """
        if test_labels_path is None:
            self.test_labels_path = os.path.join(Path().test_data, "test_label_fixed.json")
        else:
            self.test_labels_path = test_labels_path
            
        self.test_data = self._load_test_data()
        self._file_cache = {}  # Cache to store found file paths
        self._initialize_file_cache()
        
        self.camera = DummyCamera()
        self.api = API(self.camera)
        
    def _initialize_file_cache(self):
        """Initialize a cache of image files for faster lookups."""
        Logger.info("Building file cache for faster image lookups...")
        count = 0
        
        # Walk the directory and find all .jpg files
        for root, dirs, files in os.walk(Path().test_data):
            for file in files:
                if file.endswith(".jpg"):
                    # Store with the filename as key for quick lookup
                    self._file_cache[file] = os.path.join(root, file)
                    count += 1
                    
                    if count % 1000 == 0:
                        Logger.info(f"Found {count} image files so far...")
        
        Logger.info(f"File cache built with {count} images.")
        
    def _find_image_file(self, raw_file_path: str) -> str:
        """
        Find the actual image file path based on the raw path from the dataset.
        
        Args:
            raw_file_path: The raw file path from the dataset (e.g. "clips/0530/folder/20.jpg")
            
        Returns:
            The actual full path to the image file
        """
        # Extract just the filename
        filename = os.path.basename(raw_file_path)
        
        # Check if we have it in our cache
        if filename in self._file_cache:
            return self._file_cache[filename]
            
        # If not in cache, try direct path
        normalized_path = raw_file_path.replace('/', os.sep)
        direct_path = os.path.join(Path().test_data, normalized_path)
        if os.path.exists(direct_path):
            self._file_cache[filename] = direct_path
            return direct_path
            
        # Try with clips directory
        clips_path = os.path.join(Path().test_data, "clips", normalized_path.replace(f"clips{os.sep}", ""))
        if os.path.exists(clips_path):
            self._file_cache[filename] = clips_path
            return clips_path
            
        # Extract date and folder, assuming format like "clips/0530/1492627096583412101_0/20.jpg"
        parts = normalized_path.split(os.sep)
        if len(parts) >= 3:
            # Try with the last 3 parts (date/folder/file)
            last_parts = parts[-3:]
            alt_path = os.path.join(Path().test_data, "clips", *last_parts)
            if os.path.exists(alt_path):
                self._file_cache[filename] = alt_path
                return alt_path
        
        # If we still can't find it, try a manual search by filename only (SLOW!)
        Logger.warning(f"File not found in cache, searching for {filename}...")
        for root, dirs, files in os.walk(Path().test_data):
            if filename in files:
                full_path = os.path.join(root, filename)
                Logger.info(f"Found file by name: {full_path}")
                self._file_cache[filename] = full_path
                return full_path
                
        # If all else fails, log and return the normalized path for error handling
        Logger.warning(f"Could not find file with path: {raw_file_path}")
        return os.path.join(Path().test_data, "clips", normalized_path)
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data from JSON file."""
        Logger.info(f"Loading test data from {self.test_labels_path}")
        with open(self.test_labels_path, 'r') as f:
            test_data = json.load(f)
        Logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def verify_test_data(self, max_samples: int = None) -> Dict[str, Any]:
        """
        Check if test data files exist in the filesystem.
        
        Args:
            max_samples: Maximum number of samples to check (None for all)
            
        Returns:
            Dictionary with verification results
        """
        if max_samples is not None:
            samples = self.test_data[:max_samples]
        else:
            samples = self.test_data
        
        found = 0
        missing = 0
        missing_files = []
        
        Logger.info(f"Verifying {len(samples)} test data files...")
        
        for i, sample in enumerate(samples):
            image_path = self._find_image_file(sample["raw_file"])
            
            if os.path.exists(image_path):
                found += 1
            else:
                missing += 1
                missing_files.append(sample["raw_file"])
                
            if i % 50 == 0 and i > 0:
                Logger.info(f"Checked {i}/{len(samples)} files. Found: {found}, Missing: {missing}")
        
        Logger.info(f"Verification complete. Found: {found}, Missing: {missing}")
        
        if missing > 0:
            Logger.warning(f"Missing {missing} files. First few: {missing_files[:5]}")
            
        return {
            "total": len(samples),
            "found": found,
            "missing": missing,
            "missing_files": missing_files
        }
    
    def _convert_tusimple_to_mask(self, lanes: List[List[int]], h_samples: List[int]) -> np.ndarray:
        """
        Convert TuSimple lane format to binary segmentation mask.
        
        Args:
            lanes: List of lane x-coordinates for each h_sample (-2 means no lane at that position)
            h_samples: List of y-coordinates for sampling
            
        Returns:
            Binary mask where lanes are marked with 1s
        """
        # Create an empty mask with model's default size
        mask = np.zeros(LaneDetector.DEFAULT_IMAGE_SIZE[::-1], dtype=np.uint8)
        
        # Process each lane
        for lane in lanes:
            points = []
            for i, h in enumerate(h_samples):
                if i < len(lane) and lane[i] != -2:  # -2 means no lane point
                    x = int(lane[i])
                    y = int(h)
                    
                    # Scale coordinates to model's image size
                    if x >= 0:  # Ensure valid x-coordinate
                        x = int(x * LaneDetector.DEFAULT_IMAGE_SIZE[0] / 1280)
                        y = int(y * LaneDetector.DEFAULT_IMAGE_SIZE[1] / 720)
                        points.append((x, y))
            
            # Draw line between points to create a continuous lane
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(mask, points[i], points[i+1], 1, 5)
                    
        return mask
    
    def _extract_individual_gt_lanes(self, lanes: List[List[int]], h_samples: List[int]) -> List[np.ndarray]:
        """
        Extract individual lane masks from TuSimple ground truth.
        
        Args:
            lanes: List of lane x-coordinates
            h_samples: List of y-coordinates
            
        Returns:
            List of binary masks, one for each ground truth lane
        """
        gt_lanes = []
        
        for lane in lanes:
            # Skip lanes with no valid points (all -2)
            if all(x == -2 for x in lane):
                continue
                
            lane_mask = np.zeros(LaneDetector.DEFAULT_IMAGE_SIZE[::-1], dtype=np.uint8)
            points = []
            
            for i, h in enumerate(h_samples):
                if i < len(lane) and lane[i] != -2:
                    x = int(lane[i])
                    y = int(h)
                    
                    # Scale coordinates to model's image size
                    if x >= 0:
                        x = int(x * LaneDetector.DEFAULT_IMAGE_SIZE[0] / 1280)
                        y = int(y * LaneDetector.DEFAULT_IMAGE_SIZE[1] / 720)
                        points.append((x, y))
            
            # Draw the lane on the mask
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(lane_mask, points[i], points[i+1], 1, 5)
            
            # Only add non-empty masks
            if np.any(lane_mask):
                gt_lanes.append(lane_mask)
        
        return gt_lanes
    
    def _extract_lane_masks_from_instances(self, instances_map: torch.Tensor) -> List[np.ndarray]:
        """
        Extract individual lane masks from the model's instance segmentation map.
        
        Args:
            instances_map: Instance segmentation map from the lane detector
            
        Returns:
            List of binary masks, one for each detected lane
        """
        if instances_map is None:
            return []

        # Get unique lane IDs (0 is background, so exclude it)
        unique_instances = torch.unique(instances_map)
        unique_instances = unique_instances[unique_instances != 0]

        lane_masks = []
        for instance in unique_instances:
            # Create a binary mask for this lane instance
            mask = (instances_map == instance).cpu().numpy().astype(np.uint8)
            lane_masks.append(mask)

        return lane_masks
    
    def calculate_iou(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate Intersection over Union between prediction and ground truth.
        
        Args:
            prediction: Binary mask of predicted lanes
            ground_truth: Binary mask of ground truth lanes
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection (pixels where both masks are 1)
        intersection = np.logical_and(prediction, ground_truth).sum()
        
        # Calculate union (pixels where either mask is 1)
        union = np.logical_or(prediction, ground_truth).sum()
        
        # Handle edge case of empty masks
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def calculate_per_lane_confusion_matrix(self, predicted_lanes: List[np.ndarray], 
                                           ground_truth_lanes: List[np.ndarray]) -> Dict[str, int]:
        """
        Calculate confusion matrix for lane detection on a per-lane basis.
        
        Args:
            predicted_lanes: List of binary masks for each predicted lane
            ground_truth_lanes: List of binary masks for each ground truth lane
            
        Returns:
            Dictionary with TP, FP, FN counts
        """
        # If either predicted or ground truth lanes are empty, handle accordingly
        if not predicted_lanes and not ground_truth_lanes:
            return {"tp": 0, "fp": 0, "fn": 0}
        elif not predicted_lanes:
            return {"tp": 0, "fp": 0, "fn": len(ground_truth_lanes)}
        elif not ground_truth_lanes:
            return {"tp": 0, "fp": len(predicted_lanes), "fn": 0}
        
        # Calculate IoU between each predicted lane and each ground truth lane
        iou_matrix = np.zeros((len(predicted_lanes), len(ground_truth_lanes)))
        
        for i, pred_lane in enumerate(predicted_lanes):
            for j, gt_lane in enumerate(ground_truth_lanes):
                iou = self.calculate_iou(pred_lane, gt_lane)
                iou_matrix[i, j] = iou
        
        # Define IoU threshold for matching
        iou_threshold = 0.35  # Consider a lane as matching if IoU > threshold
        
        # Apply Hungarian algorithm to find optimal matching that maximizes total IoU
        # First, create a cost matrix (1 - IoU for valid matches, inf for invalid)
        cost_matrix = np.ones_like(iou_matrix) * float('inf')
        valid_matches = iou_matrix > iou_threshold
        cost_matrix[valid_matches] = 1.0 - iou_matrix[valid_matches]
        
        try:
            # Find optimal assignment (minimum cost, maximum IoU)
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Count true positives (valid assignments)
            tp = sum(cost_matrix[row_indices[i], col_indices[i]] != float('inf') 
                    for i in range(len(row_indices)))
            
            # False positives: predicted lanes that weren't matched
            fp = len(predicted_lanes) - tp
            
            # False negatives: ground truth lanes that weren't matched
            fn = len(ground_truth_lanes) - tp
            
        except Exception as e:
            # Fallback if Hungarian algorithm fails
            Logger.warning(f"Hungarian algorithm failed: {str(e)}")
            
            # Use greedy matching instead
            matched_pred = set()
            matched_gt = set()
            
            tp = 0
            
            # For each GT lane, find the best predicted lane
            for j in range(len(ground_truth_lanes)):
                if j < iou_matrix.shape[1]:  # Safety check
                    best_pred = -1
                    best_iou = iou_threshold
                    
                    for i in range(len(predicted_lanes)):
                        if i < iou_matrix.shape[0] and i not in matched_pred:  # Safety check
                            if iou_matrix[i, j] > best_iou:
                                best_iou = iou_matrix[i, j]
                                best_pred = i
                    
                    if best_pred >= 0:
                        tp += 1
                        matched_pred.add(best_pred)
                        matched_gt.add(j)
            
            fp = len(predicted_lanes) - tp
            fn = len(ground_truth_lanes) - tp
        
        return {"tp": tp, "fp": fp, "fn": fn}
        
    def evaluate_single_image(self, image_path: str, gt_data: Dict) -> Dict[str, Any]:
        """
        Evaluate metrics for a single image.
        
        Args:
            image_path: Path to the image file
            gt_data: Ground truth data for the image
            
        Returns:
            Dictionary with calculated metrics
        """
        # Get the actual file path
        image_path = self._find_image_file(image_path)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            Logger.error(f"Failed to load image: {image_path}")
            return {"error": f"Failed to load image: {image_path}"}

        # Run the lane detector
        instances_map = self.api._detector(image)
        
        # Handle case where no lanes are detected
        if instances_map is None:
            Logger.warning(f"No lanes detected in {image_path}")
            return {
                "iou": 0.0,
                "confusion_matrix": {"tp": 0, "fp": 0, "fn": len(gt_data["lanes"])},
                "num_pred_lanes": 0,
                "num_gt_lanes": len(gt_data["lanes"])
            }

        # Convert predictions to binary mask (all lanes combined)
        pred_binary = (instances_map > 0).cpu().numpy().astype(np.uint8)
        
        # Convert ground truth to binary mask (all lanes combined)
        gt_binary = self._convert_tusimple_to_mask(gt_data["lanes"], gt_data["h_samples"])
        
        # Calculate overall IoU
        iou = self.calculate_iou(pred_binary, gt_binary)
        
        # Extract individual lane masks from predictions
        pred_lanes = self._extract_lane_masks_from_instances(instances_map)
        
        # Extract individual lane masks from ground truth
        gt_lanes = self._extract_individual_gt_lanes(gt_data["lanes"], gt_data["h_samples"])
        
        # Calculate per-lane confusion matrix
        conf_matrix = self.calculate_per_lane_confusion_matrix(pred_lanes, gt_lanes)
        
        return {
            "iou": iou,
            "confusion_matrix": conf_matrix,
            "num_pred_lanes": len(pred_lanes),
            "num_gt_lanes": len(gt_lanes),
            "image": image,
            "pred_binary": pred_binary,
            "gt_binary": gt_binary,
            "instances_map": instances_map
        }
    
    def visualize_prediction(self, image_path: str, gt_data: Dict, output_path: str = None) -> np.ndarray:
        """
        Visualize the model's lane predictions alongside ground truth lanes.
        
        Args:
            image_path: Path to the image file
            gt_data: Ground truth data for the image
            output_path: Path to save the visualization (None to just return)
            
        Returns:
            Visualization image with predictions and ground truth
        """
        # Evaluate metrics for the image
        result = self.evaluate_single_image(image_path, gt_data)
        
        if "error" in result:
            Logger.error(f"Error visualizing {image_path}: {result['error']}")
            return None
            
        # Get the original image
        image = result["image"]
        
        # Resize image to model's default size
        image = cv2.resize(image, LaneDetector.DEFAULT_IMAGE_SIZE)
        
        # Create visualization image
        vis_image = image.copy()
        
        # Create predicted lanes overlay
        pred_overlay = np.zeros_like(image)
        if result["instances_map"] is not None:
            # Get individual lanes for colored visualization
            pred_lanes = self._extract_lane_masks_from_instances(result["instances_map"])
            
            # Assign different colors to each lane
            colors = [
                (0, 0, 255),   # Red
                (0, 255, 0),   # Green
                (255, 0, 0),   # Blue
                (0, 255, 255), # Yellow
                (255, 0, 255), # Magenta
                (255, 255, 0)  # Cyan
            ]
            
            for i, lane_mask in enumerate(pred_lanes):
                color = colors[i % len(colors)]
                pred_overlay[lane_mask > 0] = color
        
        # Create ground truth lanes overlay (white)
        gt_overlay = np.zeros_like(image)
        gt_binary = result["gt_binary"]
        gt_overlay[gt_binary > 0] = (255, 255, 255)
        
        # Blend overlays with original image
        alpha_pred = 0.6  # Transparency for predictions
        alpha_gt = 0.4    # Transparency for ground truth
        
        # Combine image with prediction and ground truth overlays
        vis_image = cv2.addWeighted(vis_image, 1.0, pred_overlay, alpha_pred, 0)
        vis_image = cv2.addWeighted(vis_image, 1.0, gt_overlay, alpha_gt, 0)
        
        # Add metrics as text
        iou_text = f"IoU: {result['iou']:.4f}"
        tp_text = f"TP: {result['confusion_matrix']['tp']}"
        fp_text = f"FP: {result['confusion_matrix']['fp']}"
        fn_text = f"FN: {result['confusion_matrix']['fn']}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Add background rectangle for better text visibility
        for i, text in enumerate([iou_text, tp_text, fp_text, fn_text]):
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(vis_image, (10, 20 + i*25 - 15), (10 + text_width, 20 + i*25 + 5), bg_color, -1)
            cv2.putText(vis_image, text, (10, 20 + i*25), font, font_scale, text_color, font_thickness)
        
        # Add legend
        legend_texts = ["Predicted Lanes", "Ground Truth"]
        legend_colors = [(0, 0, 255), (255, 255, 255)]
        
        for i, (text, color) in enumerate(zip(legend_texts, legend_colors)):
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(vis_image, (vis_image.shape[1] - 10 - text_width - 30, 20 + i*25 - 15), 
                         (vis_image.shape[1] - 10, 20 + i*25 + 5), bg_color, -1)
            cv2.putText(vis_image, text, (vis_image.shape[1] - 10 - text_width - 20, 20 + i*25), 
                       font, font_scale, text_color, font_thickness)
            cv2.rectangle(vis_image, (vis_image.shape[1] - 10 - 15, 20 + i*25 - 10), 
                         (vis_image.shape[1] - 10, 20 + i*25), color, -1)
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            Logger.info(f"Visualization saved to {output_path}")
            
        return vis_image
    
    def evaluate_dataset(self, num_samples: int = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate lane detection metrics on multiple images from the dataset.
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            output_dir: Directory to save visualizations (None for no saving)
            
        Returns:
            Dictionary with aggregated metrics
        """
        if num_samples is not None:
            samples = self.test_data[:num_samples]
        else:
            samples = self.test_data
            
        Logger.info(f"Evaluating metrics on {len(samples)} images...")
        
        # Initialize aggregated metrics
        total_iou = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        num_valid_samples = 0
        
        # Set up progress tracking
        total_samples = len(samples)
        progress_interval = max(1, total_samples // 20)  # Show progress at 5% intervals
        
        # Process each sample
        for i, sample in enumerate(samples):
            image_path = sample["raw_file"]
            
            # Evaluate metrics for this image
            result = self.evaluate_single_image(image_path, sample)
            
            # Skip samples with errors
            if "error" in result:
                continue
                
            # Update aggregated metrics
            total_iou += result["iou"]
            total_tp += result["confusion_matrix"]["tp"]
            total_fp += result["confusion_matrix"]["fp"]
            total_fn += result["confusion_matrix"]["fn"]
            num_valid_samples += 1
            
            # Save visualization if requested
            if output_dir:
                # Create a filename based on the original path
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"vis_{base_name}")
                self.visualize_prediction(image_path, sample, output_path)
            
            # Show progress
            if (i + 1) % progress_interval == 0 or (i + 1) == total_samples:
                progress = (i + 1) / total_samples * 100
                Logger.info(f"Progress: {progress:.1f}% ({i + 1}/{total_samples})")
        
        # Calculate average metrics
        if num_valid_samples > 0:
            avg_iou = total_iou / num_valid_samples
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            avg_iou = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        # Compile and return results
        results = {
            "num_samples": total_samples,
            "num_valid_samples": num_valid_samples,
            "avg_iou": avg_iou,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Print summary
        Logger.info("Evaluation complete. Summary:")
        Logger.info(f"Average IoU: {avg_iou:.4f}")
        Logger.info(f"Precision: {precision:.4f}")
        Logger.info(f"Recall: {recall:.4f}")
        Logger.info(f"F1 Score: {f1:.4f}")
        
        return results
        
    def print_metrics_summary(self, results: Dict[str, Any]):
        """
        Print a formatted summary of evaluation metrics.
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "="*50)
        print("LANE DETECTION EVALUATION METRICS SUMMARY")
        print("="*50)
        print(f"Total samples evaluated: {results['num_samples']}")
        print(f"Valid samples: {results['num_valid_samples']}")
        print("-"*50)
        print("Overall Segmentation Quality:")
        print(f"  Average IoU: {results['avg_iou']:.4f}")
        print("-"*50)
        print("Per-Lane Detection Quality:")
        print(f"  True Positives: {results['total_tp']}")
        print(f"  False Positives: {results['total_fp']}")
        print(f"  False Negatives: {results['total_fn']}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print("="*50)
        
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate lane detection metrics on TuSimple dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="Directory to save visualizations")
    parser.add_argument("--test_labels", type=str, default=None, help="Path to test labels JSON file")
    parser.add_argument("--verify_only", action="store_true", help="Only verify test data without evaluation")
    
    args = parser.parse_args()
    
    metrics = LaneMetrics(test_labels_path=args.test_labels)
    
    if args.verify_only:
        metrics.verify_test_data()
    else:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
        results = metrics.evaluate_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
        
        metrics.print_metrics_summary(results)
