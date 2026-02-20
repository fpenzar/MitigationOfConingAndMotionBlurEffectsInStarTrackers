import numpy as np
from degradation import Degradation
from image_operator import ImageOperator
from centroids import Centroids


class RL:

    def __init__(self, degradation: Degradation, iterations=25):
        self.degradation = degradation
        self.iterations = iterations
        self.max_match_dist = 5.0
        self.eps_frac = 0.02
        self.eps_min = 1e-3
        self.centroid_threshold = 0.4
        self.min_stars = 10
        self.require_all = False
    
    def restore(self, image, coupled=False):
        g = image
        f = g.copy().astype(np.float32)
        for i in range(self.iterations):
            print(f"    {i + 1}/{self.iterations}", end="\r")
            forward = self.degradation.blur(f, std=0, coupled=coupled)
            ratio = g / (forward + 1e-8)
            backproj = self.deblur(ratio, coupled)
            f = f * backproj
        print("")
        return f
    
    def restore_with_automatic_termination(self, image, coupled=False):
        g = np.asarray(image, dtype=np.float32)
        f = np.clip(g.copy(), 0.0, 1.0)
        centroids_prev_k = None
        centroids_current_k = None
        Dx, Dy = self.estimate_blur_lengths()
        eps_x = max(self.eps_min, self.eps_frac * max(Dx, 1.0))
        eps_y = max(self.eps_min, self.eps_frac * max(Dy, 1.0))

        for k in range(3*self.iterations):
            print(f"    {k + 1}", end="\r")
            forward = self.degradation.blur(f, std=0, coupled=coupled)
            ratio = g / (forward + 1e-8)
            backproj = self.deblur(ratio, coupled=coupled)
            f = np.clip(f * backproj, 0.0, 1.0)
            centroids_next_k = Centroids.compute_centroids(f, threshold=self.centroid_threshold)

            # Calculate the auto termination condition
            if centroids_prev_k is not None and centroids_current_k is not None:
                # match k-1 -> k and k -> k+1
                pairs1 = Centroids.match_centroids(centroids_prev_k, centroids_current_k, max_dist=self.max_match_dist)
                pairs2 = Centroids.match_centroids(centroids_current_k, centroids_next_k, max_dist=self.max_match_dist)

                # enough stars need to match
                if len(pairs1) < self.min_stars or len(pairs2) < self.min_stars:
                    pass
                else:
                    prev_centroid_at_k = {tuple(centroid_k): centroid_k_minus_1 for centroid_k_minus_1, centroid_k in pairs1}
                    next_centroid_at_k = {tuple(centroid_k): centroid_k_plus_1 for centroid_k, centroid_k_plus_1 in pairs2}
                    stable_stars = []

                    for centroid_k_tuple, centroid_k_minus_1 in prev_centroid_at_k.items():
                        # Only consider stars tracked in all three iterations
                        if centroid_k_tuple not in next_centroid_at_k:
                            continue

                        centroid_k = np.array(centroid_k_tuple)
                        centroid_prev_k = np.array(centroid_k_minus_1)
                        centroid_next_k = np.array(next_centroid_at_k[centroid_k_tuple])

                        # Step sizes between iterations
                        step_x_k = abs(centroid_next_k[1] - centroid_k[1])
                        step_x_km1 = abs(centroid_k[1] - centroid_prev_k[1])
                        step_y_k = abs(centroid_next_k[0] - centroid_k[0])
                        step_y_km1 = abs(centroid_k[0] - centroid_prev_k[0])

                        # Termination condition from the paper
                        x_stable = abs(step_x_k - step_x_km1) <= eps_x
                        y_stable = abs(step_y_k - step_y_km1) <= eps_y

                        stable_stars.append(x_stable and y_stable)

                    # Decide whether to stop
                    if len(stable_stars) >= self.min_stars:
                        if self.require_all:
                            should_stop = all(stable_stars)
                        else:
                            should_stop = (np.mean(stable_stars) > 0.8)

                        if should_stop:
                            break

            centroids_prev_k, centroids_current_k = centroids_current_k, centroids_next_k
        print("")
        return f
    

    def deblur(self, image, coupled=False):
        z_deblurred = ImageOperator.scatter(image, self.degradation.z_kernels_forwards, self.degradation.max_dim)
        if not coupled:
            xyz_deblurred = ImageOperator.correlate(z_deblurred, self.degradation.xy_kernel)
        else:
            xyz_deblurred = z_deblurred
        return xyz_deblurred
    

    def estimate_blur_lengths(self):
        if self.degradation.xy_kernel.shape == (1,):
            return 1, 1
        k = np.asarray(self.degradation.xy_kernel, dtype=np.float32)
        ys, xs = np.where(k > 0)
        if len(xs) == 0:
            return 0.0, 0.0
        dx = xs.max() - xs.min()
        dy = ys.max() - ys.min()
        return float(dx), float(dy)

