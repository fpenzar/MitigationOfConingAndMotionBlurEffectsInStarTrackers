import numpy as np
from scipy.ndimage import label
import cv2


class Centroids:

    def __init__(self):
        pass

    @classmethod
    def star_within_image_center(cls, y0, y1, x0, x1, h, w):
        threshold = 30
        if y0 < threshold or x0 < threshold:
            return False
        if (h-y1) < threshold or (w-x1) < threshold:
            return False
        return True
    
    @classmethod
    def compute_centroids(cls, image, threshold, plot=False, method_name=""):
        H, W = image.shape
        binary = image > threshold
        labels, num = label(binary)

        if plot:
            binary_vis = (binary.astype(np.uint8)) * 255
            cv2.imshow(f"{method_name} binary", binary_vis)

        centroids = []
        for star_id in range(1, num+1):
            star_y, star_x = np.where(labels == star_id)
            # to small for a star or too big because it is an artifact
            if len(star_x) < 3 or len(star_x) > 50:
                continue 
            y0, y1 = star_y.min(), star_y.max()+1
            x0, x1 = star_x.min(), star_x.max()+1
            if not cls.star_within_image_center(y0, y1, x0, x1, H, W):
                continue
            
            patch = image[y0:y1, x0:x1]
            star  = (labels[y0:y1, x0:x1] == star_id)
            patch_masked = patch * star

            # Calculate the star image center
            h, w = patch.shape
            yy, xx = np.mgrid[0:h, 0:w]
            total = patch_masked.sum()
            cx = (patch_masked * xx).sum() / total
            cy = (patch_masked * yy).sum() / total
            
            # Offset to image coordinates
            full_x = x0 + cx
            full_y = y0 + cy
            centroids.append((full_y, full_x))
        return centroids
    

    @classmethod
    def distance_between_centroids(cls, centroid_1, centroid_2):
        y1, x1 = centroid_1
        y2, x2 = centroid_2
        return np.sqrt((y1-y2)**2 + (x1-x2)**2)
    
    @classmethod
    def find_closest_centroids(cls, centroids_estimated, centroids_truth):
        min_distances = []
        for estimated_centroid in centroids_estimated:
            corresponding_centroid = (-1, -1)
            min_distance = np.inf
            for truth_centroid in centroids_truth:
                distance = cls.distance_between_centroids(estimated_centroid, truth_centroid)
                if distance < min_distance:
                    min_distance = distance
                    corresponding_centroid = truth_centroid
            min_distances.append(min_distance)
            # print(f"{estimated_centroid} ~ {corresponding_centroid}")
        
        if len(min_distances) > 3:
            return np.mean(np.asarray(min_distances))
        else:
            return np.nan


    @classmethod
    def evaluate(cls, ground_truth_image, estimated_image, threshold_gt, threshold_est, method_name=None):
        gt_centroids = cls.compute_centroids(ground_truth_image, threshold_gt)
        est_centroids = cls.compute_centroids(estimated_image, threshold_est, plot=True, method_name=method_name)
        if len(est_centroids) < 3:
            mean_distance = np.nan
        else:
            mean_distance = cls.find_closest_centroids(est_centroids, gt_centroids)
        if method_name is not None:
            print(f"{method_name} mean distance: {mean_distance} pixels, {len(est_centroids)} stars identified")
        return mean_distance
    
    @classmethod
    def match_centroids(cls, previous, current, max_dist=5.0):
        if len(previous) == 0 or len(current) == 0:
            return []
        prev_used = set()
        pairs = []
        for c in current:
            best = None
            lowest_distance = np.inf
            best_index = None
            for j, p in enumerate(previous):
                if j in prev_used:
                    continue
                distance = cls.distance_between_centroids(c, p)
                if distance < lowest_distance:
                    lowest_distance = distance
                    best = p
                    best_index = j
            if best is not None and lowest_distance <= max_dist:
                prev_used.add(best_index)
                pairs.append((best, c))
        return pairs