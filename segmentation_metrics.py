#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import cv2
import tensorflow as tf
import tqdm
import loguru
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
import scipy.ndimage
import json

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_segmentation_metrics')

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union between prediction and ground truth masks
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    return intersection / union

def calculate_precision_recall(pred_mask, gt_mask):
    """
    Calculate precision and recall for binary segmentation
    """
    true_positives = np.logical_and(pred_mask, gt_mask).sum()
    false_positives = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    false_negatives = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def calculate_containment(pred_mask, gt_mask, threshold=0.5):
    """
    For each connected component (lane) in the ground truth, compute the fraction of its pixels covered by the prediction.
    If this fraction is above threshold, count as correct. Return the fraction of lanes that are contained.
    """
    # Label connected components in gt_mask
    labeled_gt, num_lanes = scipy.ndimage.label(gt_mask > 0)
    if num_lanes == 0:
        return 1.0  # If no lanes, consider as perfect containment
    contained = 0
    for lane_id in range(1, num_lanes+1):
        lane_mask = (labeled_gt == lane_id)
        if lane_mask.sum() == 0:
            continue
        covered = np.logical_and(lane_mask, pred_mask > 0).sum() / float(lane_mask.sum())
        if covered >= threshold:
            contained += 1
    return contained / float(num_lanes)

def calculate_lane_level_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Lane-level (containment-aware) precision, recall, F1.
    For each GT lane, if at least threshold of its pixels are covered by the prediction, count as TP.
    For each predicted lane, if it does not match any GT lane, count as FP.
    """
    # Label connected components in gt_mask and pred_mask
    labeled_gt, num_gt = scipy.ndimage.label(gt_mask > 0)
    labeled_pred, num_pred = scipy.ndimage.label(pred_mask > 0)
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0  # perfect
    if num_gt == 0:
        return 0.0, 1.0, 0.0
    if num_pred == 0:
        return 1.0, 0.0, 0.0
    # For each GT lane, check if contained in any pred lane
    TP = 0
    for gt_id in range(1, num_gt+1):
        gt_lane = (labeled_gt == gt_id)
        best_covered = 0
        for pred_id in range(1, num_pred+1):
            pred_lane = (labeled_pred == pred_id)
            covered = np.logical_and(gt_lane, pred_lane).sum() / float(gt_lane.sum())
            if covered > best_covered:
                best_covered = covered
        if best_covered >= threshold:
            TP += 1
    FN = num_gt - TP
    # For each pred lane, check if it matches any GT lane
    FP = 0
    for pred_id in range(1, num_pred+1):
        pred_lane = (labeled_pred == pred_id)
        best_covered = 0
        for gt_id in range(1, num_gt+1):
            gt_lane = (labeled_gt == gt_id)
            covered = np.logical_and(gt_lane, pred_lane).sum() / float(gt_lane.sum())
            if covered > best_covered:
                best_covered = covered
        if best_covered < threshold:
            FP += 1
    precision = TP / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_segmentation(src_dir, weights_path, save_dir, num_images=-1):
    """
    Evaluate LaneNet segmentation performance
    Args:
        src_dir: Source directory containing test images and ground truth
        weights_path: Path to model weights
        save_dir: Directory to save evaluation results
        num_images: Number of images to evaluate (-1 for all)
    """
    # Initialize model
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    
    # Initialize postprocessor
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    
    # Create session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    
    # Initialize metrics
    total_lane_prec = 0
    total_lane_rec = 0
    total_lane_f1 = 0
    processed_images = 0
    num_images_with_lanes = 0
    per_hundred_results = []
    hundred_lane_prec = 0
    hundred_lane_rec = 0
    hundred_lane_f1 = 0
    hundred_count = 0
    
    # Check for checkpoint and load progress if needed
    checkpoint_path = None
    start_image_idx = 0
    if num_images == -1 and save_dir:
        checkpoint_path = os.path.join(save_dir, 'metrics_results.json')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as jf:
                checkpoint_data = json.load(jf)
                per_hundred_results = checkpoint_data.get('per_hundred', [])
                start_image_idx = len(per_hundred_results) * 100
    
    with tf.Session(config=sess_config) as sess:
        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
        
        # define saver
        saver = tf.train.Saver(variables_to_restore)
        
        saver.restore(sess=sess, save_path=weights_path)
        
        # Determine if we're using training or testing data
        if 'train_set' in src_dir:
            data_type = 'training'
            file_name = 'train.txt'
        elif 'test_set' in src_dir:
            data_type = 'testing'
            file_name = 'test.txt'
        else:
            raise ValueError('Source directory must contain either "train_set" or "test_set" in its path')

        # Read test.txt file
        test_file = os.path.join(src_dir, data_type, file_name)
        assert os.path.exists(test_file), '{:s} not exist'.format(test_file)

        with open(test_file, 'r') as file:
            lines = file.readlines()
            # Skip already processed images if resuming
            lines = lines[start_image_idx:]
            for line in tqdm.tqdm(lines):
                # Check if we've processed enough images
                if num_images != -1 and processed_images >= num_images:
                    break

                image_path, binary_gt_path, instance_gt_path = line.strip().split(' ')

                # Read images
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                binary_gt = cv2.imread(binary_gt_path, cv2.IMREAD_GRAYSCALE)
                instance_gt = cv2.imread(instance_gt_path, cv2.IMREAD_GRAYSCALE)

                if image is None or binary_gt is None or instance_gt is None:
                    LOG.error('Failed to read image pair: {:s}'.format(image_path))
                    continue

                # Store original image dimensions
                original_height, original_width = image.shape[:2]

                # Prepare input image
                image_vis = image
                image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0
            
                # Run inference
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                
                # Post-process results
                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis,
                    with_lane_fit=False,  # We don't need curve fitting
                    data_source='tusimple'
                )
                
                # Create binary mask from postprocessed result
                pred_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                if postprocess_result['mask_image'] is not None:
                    mask_image = postprocess_result['mask_image']
                    # Convert to binary mask
                    pred_mask = np.any(mask_image > 0, axis=2).astype(np.uint8) * 255
                    pred_mask = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                
                # Calculate metrics
                iou = calculate_iou(pred_mask > 0, binary_gt > 0)
                precision, recall = calculate_precision_recall(pred_mask > 0, binary_gt > 0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                containment = calculate_containment(pred_mask, binary_gt, threshold=0.5)
                lane_prec, lane_rec, lane_f1 = calculate_lane_level_metrics(pred_mask, binary_gt, threshold=0.5)
                
                # Update metrics
                total_lane_prec += lane_prec
                total_lane_rec += lane_rec
                total_lane_f1 += lane_f1
                processed_images += 1
                hundred_lane_prec += lane_prec
                hundred_lane_rec += lane_rec
                hundred_lane_f1 += lane_f1
                hundred_count += 1
                # If resuming, add to totals from checkpoint
                if start_image_idx > 0:
                    processed_images += start_image_idx
                    start_image_idx = 0
                
                # Save per-hundred block results
                if hundred_count == 100:
                    per_hundred_results.append({
                        'block': len(per_hundred_results) + 1,
                        'lane_precision': hundred_lane_prec / hundred_count,
                        'lane_recall': hundred_lane_rec / hundred_count,
                        'lane_f1': hundred_lane_f1 / hundred_count
                    })
                    hundred_lane_prec = 0
                    hundred_lane_rec = 0
                    hundred_lane_f1 = 0
                    hundred_count = 0
                    # Save results to JSON after each 100 block
                    if save_dir:
                        avg_lane_prec_so_far = (total_lane_prec + sum([b['lane_precision'] for b in per_hundred_results]) * 100) / processed_images
                        avg_lane_rec_so_far = (total_lane_rec + sum([b['lane_recall'] for b in per_hundred_results]) * 100) / processed_images
                        avg_lane_f1_so_far = (total_lane_f1 + sum([b['lane_f1'] for b in per_hundred_results]) * 100) / processed_images
                        results = {
                            'total': {
                                'lane_precision': avg_lane_prec_so_far,
                                'lane_recall': avg_lane_rec_so_far,
                                'lane_f1': avg_lane_f1_so_far,
                                'num_images': processed_images
                            },
                            'per_hundred': per_hundred_results
                        }
                        json_path = os.path.join(save_dir, 'metrics_results.json')
                        with open(json_path, 'w') as jf:
                            json.dump(results, jf, indent=2)
                
                # Log partial results every 10 images
                if processed_images % 10 == 0:
                    avg_lane_prec = total_lane_prec / processed_images
                    avg_lane_rec = total_lane_rec / processed_images
                    avg_lane_f1 = total_lane_f1 / processed_images
                    LOG.info('Processed {} images. Lane-level (containment) metrics so far: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
                        processed_images, avg_lane_prec, avg_lane_rec, avg_lane_f1))
                
                # Save visualization
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    input_image_name = os.path.split(image_path)[1]
                    base_name = os.path.splitext(input_image_name)[0]

                    # --- Overlay visualization (xxxx.png) ---
                    overlay = image_vis.copy()
                    # Create color masks
                    pred_color = np.zeros_like(overlay)
                    gt_color = np.zeros_like(overlay)
                    overlap_color = np.zeros_like(overlay)
                    # Red for pred
                    pred_color[(pred_mask > 0) & (binary_gt == 0)] = [0, 0, 255]
                    # Green for gt
                    gt_color[(binary_gt > 0) & (pred_mask == 0)] = [0, 255, 0]
                    # Yellow for overlap
                    overlap_color[(pred_mask > 0) & (binary_gt > 0)] = [0, 255, 255]
                    # Add overlays
                    overlay = cv2.addWeighted(overlay, 1.0, pred_color, 0.5, 0)
                    overlay = cv2.addWeighted(overlay, 1.0, gt_color, 0.5, 0)
                    overlay = cv2.addWeighted(overlay, 1.0, overlap_color, 0.7, 0)

                    # Draw legend (top-left corner, small box)
                    legend_h = 80
                    legend_w = 260
                    cv2.rectangle(overlay, (10, 10), (10+legend_w, 10+legend_h), (0,0,0), -1)
                    cv2.rectangle(overlay, (10, 10), (10+legend_w, 10+legend_h), (255,255,255), 2)
                    cv2.putText(overlay, 'Legend:', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.rectangle(overlay, (20, 45), (40, 65), (0,255,0), -1)
                    cv2.putText(overlay, 'GT', (45, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.rectangle(overlay, (100, 45), (120, 65), (0,0,255), -1)
                    cv2.putText(overlay, 'Pred', (125, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.rectangle(overlay, (180, 45), (200, 65), (0,255,255), -1)
                    cv2.putText(overlay, 'Overlap', (205, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    # Draw metrics (top-right corner)
                    # metrics_text = 'IoU: {:.3f}'.format(iou)  # Remove IoU
                    metrics_text2 = 'LanePrec: {:.3f}  LaneRec: {:.3f}  LaneF1: {:.3f}'.format(lane_prec, lane_rec, lane_f1)
                    (text_w2, text_h2), _ = cv2.getTextSize(metrics_text2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    x_metrics2 = overlay.shape[1] - text_w2 - 20
                    y_metrics2 = 65
                    cv2.rectangle(overlay, (x_metrics2-10, 45), (x_metrics2+text_w2+10, 45+text_h2+20), (0,0,0), -1)
                    cv2.rectangle(overlay, (x_metrics2-10, 45), (x_metrics2+text_w2+10, 45+text_h2+20), (255,255,255), 2)
                    cv2.putText(overlay, metrics_text2, (x_metrics2, y_metrics2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    overlay_path = os.path.join(save_dir, "{}.png".format(base_name))
                    cv2.imwrite(overlay_path, overlay)

                    # --- Side-by-side masks visualization (xxxx_masks.png) ---
                    pred_vis = np.stack([pred_mask]*3, axis=-1)
                    gt_vis = np.stack([binary_gt]*3, axis=-1)
                    masks_side = np.hstack((pred_vis, gt_vis))
                    # Add metrics text at the top center
                    # metrics_text_pix = 'IoU: {:.3f}'.format(iou)  # Remove IoU
                    metrics_text_lane = 'LanePrec: {:.3f}  LaneRec: {:.3f}  LaneF1: {:.3f}'.format(lane_prec, lane_rec, lane_f1)
                    (text_w_lane, text_h_lane), _ = cv2.getTextSize(metrics_text_lane, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    total_h = text_h_lane + 40
                    masks_side_with_text = np.zeros((masks_side.shape[0]+total_h, masks_side.shape[1], 3), dtype=np.uint8)
                    masks_side_with_text[total_h:,:,:] = masks_side
                    cv2.putText(masks_side_with_text, metrics_text_lane, ((masks_side_with_text.shape[1]-text_w_lane)//2, text_h_lane+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    # Add small legends below each mask
                    cv2.putText(masks_side_with_text, 'Prediction', (masks_side.shape[1]//4-60, masks_side_with_text.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.putText(masks_side_with_text, 'Ground Truth', (3*masks_side.shape[1]//4-60, masks_side_with_text.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    masks_path = os.path.join(save_dir, "{}_masks.png".format(base_name))
                    cv2.imwrite(masks_path, masks_side_with_text)
    
    # Calculate average metrics
    avg_lane_prec = total_lane_prec / processed_images
    avg_lane_rec = total_lane_rec / processed_images
    avg_lane_f1 = total_lane_f1 / processed_images
    LOG.info('Final Lane-level (containment) metrics over {} images: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
        processed_images, avg_lane_prec, avg_lane_rec, avg_lane_f1))

    if hundred_count > 0:
        # Save the last block if it has less than 100 images
        per_hundred_results.append({
            'block': len(per_hundred_results) + 1,
            'lane_precision': hundred_lane_prec / hundred_count,
            'lane_recall': hundred_lane_rec / hundred_count,
            'lane_f1': hundred_lane_f1 / hundred_count
        })
    if save_dir:
        results = {
            'total': {
                'lane_precision': total_lane_prec / processed_images,
                'lane_recall': total_lane_rec / processed_images,
                'lane_f1': total_lane_f1 / processed_images,
                'num_images': processed_images
            },
            'per_hundred': per_hundred_results
        }
        json_path = os.path.join(save_dir, 'metrics_results.json')
        with open(json_path, 'w') as jf:
            json.dump(results, jf, indent=2)

    return {
        'lane_precision': avg_lane_prec,
        'lane_recall': avg_lane_rec,
        'lane_f1': avg_lane_f1
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='Source directory containing test images and ground truth')
    parser.add_argument('--weights_path', type=str, help='Path to model weights')
    parser.add_argument('--save_dir', type=str, help='Directory to save evaluation results')
    parser.add_argument('--num_images', type=int, default=-1, help='Number of images to evaluate (-1 for all)')
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        src_dir=args.src_dir,
        weights_path=args.weights_path,
        save_dir=args.save_dir,
        num_images=args.num_images
    )