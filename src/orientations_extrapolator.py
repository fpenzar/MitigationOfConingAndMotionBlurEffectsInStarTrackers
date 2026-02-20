from gyro_integrator import GyroIntegrator
from quaternion import Quaternion
import copy
import numpy as np
from bisect import bisect_left, bisect_right


class OrientationsExtrapolator:

    DIRECT_QUATERNION=0
    CONING_APPX=1
    NOCONING=2

    def __init__(self):
        self.gyro_integrator = GyroIntegrator()
    
    def get_gt_orientations_valid_pairs(self, gt_orientations, gt_validity):
        v_mask = np.array([bool(v) for v, _ in gt_validity])
        gt_orientations = np.array(gt_orientations, dtype=object)
        valid_orientations = gt_orientations[v_mask]
        valid_pairs = []
        for i in range(len(valid_orientations) - 1):
            valid_pairs.append((valid_orientations[i], valid_orientations[i+1]))
        return valid_pairs
    
    def get_timestamps(self, sequence):
        return [t for _, t in sequence]
    
    def get_angular_velocities_for_gt_orientations(self, gt_orientation_start, gt_orientation_end, angular_velocities, angular_vel_ts):
        t_start = gt_orientation_start[1]
        t_end = gt_orientation_end[1]

        i = np.searchsorted(angular_vel_ts, t_start)
        j = np.searchsorted(angular_vel_ts, t_end)

        if j == 0:
            return []
        if i == len(angular_vel_ts) - 1:
            return []
        if j < i:
            return []
        if j == len(angular_vel_ts):
            return []
        
        angular_ts_start = angular_vel_ts[i]
        angular_ts_end = angular_vel_ts[j]
        if abs(angular_ts_start - t_start) > 0.0125 or abs(angular_ts_end - t_end) > 0.0125:
            return []
        
        output = angular_velocities[i:j]
        output.append((angular_velocities[j][0], t_end))
        return output

    
    def extrapolate(self, ground_truth_orientations, ground_truth_validity, angular_velocities, algorithm, on_valid_only=False):
        """
        Returns a list of orientations obtained by taking the valid ground truth orientation, extrapolating the orientation
        up until the next valid orientation by using the corresponding angular velocity measurements and the selected integration
        algorithm.
        """
        orientations = []
        first = True
        valid_gt_orientation_pairs = self.get_gt_orientations_valid_pairs(ground_truth_orientations, ground_truth_validity)
        angular_velocity_timestamps = self.get_timestamps(angular_velocities)

        for valid_gt_orientation_pair in valid_gt_orientation_pairs:
            gt_orientation_start, gt_orientation_end = valid_gt_orientation_pair
            matching_angular_velocities = self.get_angular_velocities_for_gt_orientations(gt_orientation_start,
                                                                                          gt_orientation_end,
                                                                                          angular_velocities,
                                                                                          angular_velocity_timestamps)
            if len(matching_angular_velocities) == 0:
                continue
            if first:
                first = False
                orientations.append(gt_orientation_start)

            init_orientation, init_start_time = gt_orientation_start
            if on_valid_only:
                num_of_samples = len(matching_angular_velocities)
                self.gyro_integrator.reset(init_orientation, init_start_time, fast_loop_samples=num_of_samples)
            else:
                self.gyro_integrator.reset(init_orientation, init_start_time, fast_loop_samples=20)

            for i, (w, t) in enumerate(matching_angular_velocities):
                if i == len(matching_angular_velocities) - 1:
                    force_orientation = True
                else:
                    force_orientation = False
                if algorithm == OrientationsExtrapolator.DIRECT_QUATERNION:
                    self.gyro_integrator.update_direct_quaternion(w, t)
                elif algorithm == OrientationsExtrapolator.CONING_APPX:
                    self.gyro_integrator.update_rotation_vector(w, t, coning=True, force_orientation=force_orientation)
                elif algorithm == OrientationsExtrapolator.NOCONING:
                    self.gyro_integrator.update_rotation_vector(w, t, coning=False, force_orientation=force_orientation)
                else:
                    raise ValueError(f"Unrecognized algorithm provided: {algorithm}")
            
            orientations.append(copy.deepcopy(self.gyro_integrator.orientations[-1]))
        
        return orientations
    

    def closest_ground_truth_orientation(self, gt_orientations, ts_gt, t):
        idx = np.searchsorted(ts_gt, t)
        if idx == 0:
            return gt_orientations[0][0]
        if idx >= len(ts_gt):
            return gt_orientations[-1][0]
        
        # choose nearer of idx-1 and idx
        if abs(ts_gt[idx] - t) > abs(t - ts_gt[idx - 1]):
            idx -= 1
        if abs(t - ts_gt[idx]) > 0.01:
            return None
        return gt_orientations[idx][0]
    

    def calculate_accumulated_error(self, ground_truth_orientations, ground_truth_validity, calculated_orientations):
        errors = []
        accumulated_error = 0
        v_mask = np.array([bool(v) for v, _ in ground_truth_validity])
        ground_truth_orientations = np.array(ground_truth_orientations, dtype=object)
        valid_orientations = ground_truth_orientations[v_mask]
        orientations_ts = self.get_timestamps(valid_orientations)
        for q_est, t in calculated_orientations:
            q_true = self.closest_ground_truth_orientation(valid_orientations, 
                                                           orientations_ts,
                                                           t)
            if q_true == None:
                if len(errors):
                    errors.append(errors[-1])
                else:
                    errors.append((0, t))
                continue
            err_angle = Quaternion.error_angle(q_est, q_true, degrees=True)
            accumulated_error += err_angle
            errors.append((accumulated_error, t))
        return errors
