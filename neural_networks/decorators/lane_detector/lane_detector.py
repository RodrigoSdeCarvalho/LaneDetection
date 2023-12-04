from neural_networks.decorators.model_decorator import ModelDecorator
from neural_networks.decorators.lane_detector.lane import Lanes
from neural_networks.decorators.lane_detector.memory import Memory
import torch
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
from typing import Optional
from utils.logger import Logger
from typing import List
import warnings
from utils.config import ImageConfig, DecoratorConfig


warnings.filterwarnings("ignore")


# Note, in numpy.where, the first index is the y-axis (0), and the second index is the x-axis (1).
class LaneDetector(ModelDecorator):
    DEFAULT_IMAGE_SIZE = (ImageConfig().preprocess_width, ImageConfig().preprocess_height)
    LANE_POINT_MASK = 1
    CENTER_POINT_MASK = 2

    def __init__(self, model, device="cuda"):
        super().__init__(model, "LaneDetector")
        if not self._model.load():
            raise Exception("Train model first")
        self._eps = 1.0
        self._device = device
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._memory = Memory()

    def __call__(self, image) -> Optional[torch.Tensor]:
        image = self._preprocess_image(image)
        binary_logits, instance_embeddings = self._model(image)
        segmentation_map = binary_logits.squeeze().argmax(dim=0)

        try:
            instances_map = self._cluster(segmentation_map, instance_embeddings)
        except:
            return self._memory.get_instances_map()

        self._memory.update(instances_map)

        return instances_map

    def _preprocess_image(self, image):
        image = cv2.resize(image, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[..., None]
        image = torch.from_numpy(image).float().permute((2, 0, 1)).unsqueeze(dim=0).to(self._device)

        return image

    def _cluster(self, segmentation_map, instance_embeddings):
        segmentation_map = segmentation_map.flatten()
        instance_embeddings = instance_embeddings.squeeze().permute(1, 2, 0).reshape(segmentation_map.shape[0], -1)
        assert segmentation_map.shape[0] == instance_embeddings.shape[0]

        mask_indices = segmentation_map.nonzero().flatten()
        cluster_data = instance_embeddings[mask_indices].detach().cpu()

        clusterer = DBSCAN(eps=self._eps)
        labels = clusterer.fit_predict(cluster_data)
        labels = torch.tensor(labels, dtype=instance_embeddings.dtype, device=self._device)

        instances_map = torch.zeros(instance_embeddings.shape[0], dtype=instance_embeddings.dtype, device=self._device)
        instances_map[mask_indices] = labels
        instances_map = instances_map.reshape(self.DEFAULT_IMAGE_SIZE[::-1])

        return instances_map

    def get_center_of_lane(self, instances_map, get_map: bool = False) -> Optional[np.ndarray]:
        lanes = self._get_segmented_lanes(instances_map)
        if lanes is None:
            cached_lanes = self._memory.get_lanes()
            if cached_lanes is None:
                return None
            lanes = cached_lanes

        center_point = lanes.get_center_point
        self._memory.update(center_point)

        # Add center point mask
        Logger.trace(f"Center point: {center_point}", show=True)

        if get_map:
            instances_map = self._apply_center_mask(instances_map, center_point)

            return instances_map
        else:
            return center_point

    def _apply_center_mask(self, instances_map, center_point) -> np.ndarray:
        instances_map = instances_map.cpu().numpy()
        instances_map[int(center_point[0]), int(center_point[1])] = LaneDetector.CENTER_POINT_MASK

        return instances_map

    def _get_segmented_lanes(self, instances_map) -> Optional[Lanes]:
        filtered_instances = self._filter_instances(instances_map)

        # If there are no lanes or only one lane, return None
        # unless there are cached lanes
        if filtered_instances is None or len(filtered_instances) < 2:
            return self._memory.get_lanes()

        center_of_image = instances_map.shape[1] / 2

        lane_left_distances = []
        lane_right_distances = []
        for lane_mask in filtered_instances:
            # Get the lane position as the mean of the x axis
            lane_position = np.mean(np.where(lane_mask > 0)[1]) if np.any(lane_mask) else None

            if lane_position is not None:
                distance_from_center = center_of_image - lane_position

                if distance_from_center < 0: # Right lane, since the distance is negative, and the coords grow from left to right
                    lane_right_distances.append((lane_mask, distance_from_center))
                else: # Left lane, since the distance is positive, and the coords grow from left to right
                    lane_left_distances.append((lane_mask, distance_from_center))

        if len(lane_left_distances) == 0 or len(lane_right_distances) == 0:
            return self._memory.get_lanes()

        # Sort the lanes by their distance from the center, getting the closest lane
        sorted_left_lanes = sorted(lane_left_distances, key=lambda x: x[1])
        sorted_right_lanes = sorted(lane_right_distances, key=lambda x: x[1], reverse=True)

        # Get the masks of the closest lanes
        left_mask, left_lane_distance = sorted_left_lanes[0]
        right_mask, right_lane_distance = sorted_right_lanes[0]

        # Check if the distance between the lanes is too big
        max_distance_threshold = DecoratorConfig().max_distance
        distance = left_lane_distance + abs(right_lane_distance)
        if distance > max_distance_threshold:
            Logger.info(f"Distance between lanes is too big: {distance}")
            return self._memory.get_lanes()

        left_mask, right_mask = self._crop_to_bev(left_mask, right_mask)
        if self._not_enough_points_in_bev_region(left_mask, right_mask):
            return self._memory.get_lanes()

        lanes = Lanes(left_mask, right_mask)
        self._memory.update(lanes)

        return lanes

    def _filter_instances(self, instances_map) -> Optional[List[np.ndarray]]:
        # Get how many unique lanes are there
        unique_instances, inverse_indices = torch.unique(instances_map.view(-1), return_inverse=True)

        # If there is only one lane, return None
        if len(unique_instances) == 1:
            return None

        # For each lane, filter the instances map unique to that lane
        lane_unique_instances = []
        for instance in unique_instances:
            if instance != 0:
                filtered_values = instances_map.clone()
                mask = instances_map != instance
                filtered_values[mask] = 0
                lane_unique_instances.append(filtered_values.cpu().numpy())

        return lane_unique_instances

    def _crop_to_bev(self, *masks) -> tuple[np.ndarray, ...]:
        cropped_masks = []
        for mask in list(masks):
            aux_mask = np.zeros_like(mask)

            min_y = ImageConfig().bev['min_y']
            max_y = ImageConfig().bev['max_y']
            min_x = ImageConfig().bev['min_x']
            max_x = ImageConfig().bev['max_x']
            aux_mask[min_y:max_y, min_x:max_x] = 1

            cropped_mask = mask * aux_mask
            cropped_masks.append(cropped_mask)

        return tuple(cropped_masks)

    def _not_enough_points_in_bev_region(self, *masks) -> bool:
        for mask in list(masks):
            if np.count_nonzero(mask) == 0:
                return True

        return False
