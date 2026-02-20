import numpy as np
from scipy.ndimage import rotate
from image_operator import ImageOperator

class Degradation:

    def __init__(self):
        self.xy_kernel = None
        self.z_kernels_forwards = None
        self.z_kernels_backwards = None
        self.wx = None
        self.wy = None
        self.wz = None
        self.T = None
        self.max_dim = None 


    def blur_with_angular_velocities(self, img_f, wx, wy, wz, T, f, pixel_size, std, p=0):
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.T = T
        self.f = None
        self.pixel_size = None
        self.h_xy_kernel(wx, wy, T, f, pixel_size)
        self.h_z_kernel(wz, T, img_f.shape)
        return self.blur(img_f, std, p)
    

    def blur(self, img_f, std, p=0, coupled=False):
        if self.xy_kernel is None or self.z_kernels_forwards is None:
            raise ValueError("Calculate the kernels before bluring!")
        
        if not coupled:
            xy_blurred = ImageOperator.convolve(img_f, self.xy_kernel)
        else:
            xy_blurred = img_f
        xyz_blurred = ImageOperator.scatter(xy_blurred, self.z_kernels_forwards, self.max_dim)
        blurred = ImageOperator.gaussian_noise(xyz_blurred, std)
        blurred = ImageOperator.salt_pepper_noise(blurred, p)
        return blurred


    def h_xy_kernel(self, wx, wy, T, f, pixel_size, kernel_size=51):
        """
        Reference: Restoration Method of a Blurred Star Image for a Star Sensor Under Dynamic Conditions
        """
        du = (f / pixel_size) * wy * T
        dv = (f / pixel_size) * -wx * T
        length = np.hypot(du, dv)
        kernel_size = int(round(max(2*length+1, kernel_size)))
        h = np.zeros((kernel_size, kernel_size), dtype=float)

        if length < 1e-6:
            h[kernel_size // 2, kernel_size // 2] = 1.0
            self.xy_kernel = h
            return h
        
        # Draw and rotate the kernel
        center = kernel_size // 2
        half = int(round(length/2))
        x0, x1 = max(0, center - half), min(kernel_size, center + half)
        h[center, x0:x1] = 1.0
        angle_deg = np.degrees(np.arctan2(dv, du))
        h = rotate(h, angle=-angle_deg, reshape=False, order=1)

        # Normalize the kernel
        total = h.sum()
        if total > 0:
            h /= total
        else:
            h[center, center] = 1.0

        self.xy_kernel = h

        return h
    
    def h_z_kernel(self, wz, T, image_shape, kernel_size=5, num_samples=200):
        H, W = image_shape
        center_y = H // 2
        center_x = W // 2
        kernels_forward = {}
        kernels_backward = {}
        max_dim = kernel_size
        phi = wz * T

        for y in range(center_y + 1):
            for x in range(center_x + 1):
                distance_y = (y - center_y)
                distance_x = (x - center_x)
                distance = np.sqrt(distance_x*distance_x + distance_y*distance_y)
                if distance < 1e-9 or abs(phi) < 1e-12:
                    kernel = np.zeros((kernel_size, kernel_size), np.float32)
                    kernel[kernel_size//2, kernel_size//2] = 1.0
                    kernels_forward[(y, x)] = (kernel, (kernel_size//2, kernel_size//2))
                    kernels_backward[(y, x)] = (kernel, (kernel_size//2, kernel_size//2))

                    kernels_forward[(H-x-1, y)] = (kernel, (kernel_size//2, kernel_size//2))
                    kernels_backward[(H-x-1, y)] = (kernel, (kernel_size//2, kernel_size//2))

                    kernels_forward[(H-y-1, W-x-1)] = (kernel, (kernel_size//2, kernel_size//2))
                    kernels_backward[(H-y-1, W-x-1)] = (kernel, (kernel_size//2, kernel_size//2))

                    kernels_forward[(x, W-y-1)] = (kernel, (kernel_size//2, kernel_size//2))
                    kernels_backward[(x, W-y-1)] = (kernel, (kernel_size//2, kernel_size//2))
                    continue

                kernel_dim = int(max(kernel_size, np.ceil(distance)*4 + 1))
                if kernel_dim > max_dim:
                    max_dim = kernel_dim

                k_backwards = np.zeros((kernel_dim, kernel_dim), np.float32)
                k_forwards = np.zeros((kernel_dim, kernel_dim), np.float32)
                kernel_center = kernel_dim // 2

                theta0 = np.arctan2(-distance_y, distance_x)

                max_x_b = 0
                min_x_b = kernel_dim
                max_y_b = 0
                min_y_b = kernel_dim
                max_x_f = 0
                min_x_f = kernel_dim
                max_y_f = 0
                min_y_f = kernel_dim
                t_values = np.linspace(-0.5, 0.5, num_samples)
                # t_values = np.linspace(0, 1, num_samples)
                for i, t in enumerate(t_values):
                    # Backwards
                    theta_backwards = theta0 + t * phi
                    ui, vi = self.calculate_kernel_coordinates(theta_backwards, distance, center_x, center_y, x, y, kernel_center)
                    if 0 <= ui < kernel_dim and 0 <= vi < kernel_dim:
                        k_backwards[vi, ui] += 1.0
                        max_x_b, min_x_b, max_y_b, min_y_b = self.determine_kernel_bounds(ui, vi, max_x_b, min_x_b, max_y_b, min_y_b)
                    
                    # Forwards
                    theta_forwards = theta0 - t * phi
                    ui, vi = self.calculate_kernel_coordinates(theta_forwards, distance, center_x, center_y, x, y, kernel_center)
                    if 0 <= ui < kernel_dim and 0 <= vi < kernel_dim:
                        k_forwards[vi, ui] += 1.0
                        max_x_f, min_x_f, max_y_f, min_y_f = self.determine_kernel_bounds(ui, vi, max_x_f, min_x_f, max_y_f, min_y_f)
                
                k_forwards = k_forwards[min_y_f:max_y_f+1, min_x_f:max_x_f+1]
                k_backwards = k_backwards[min_y_b:max_y_b+1, min_x_b:max_x_b+1]

                # Calclate the new kernel center
                cx_f = kernel_center - min_x_f
                cy_f = kernel_center - min_y_f
                cx_b = kernel_center - min_x_b
                cy_b = kernel_center - min_y_b
                H_f = max_y_f - min_y_f + 1
                W_f = max_x_f - min_x_f + 1
                H_b = max_y_b - min_y_b + 1
                W_b = max_x_b - min_x_b + 1

                # Normalize
                total = k_forwards.sum()
                if total > 0:
                    k_forwards /= total
                total = k_backwards.sum()
                if total > 0:
                    k_backwards /= total

                # 2nd quadrant
                kernels_forward[(y, x)] = (k_forwards, (cy_f, cx_f))
                kernels_backward[(y, x)] = (k_backwards, (cy_b, cx_b))

                # 3rd quadrant
                kernels_forward[(H-x-1, y)] = (np.rot90(k_forwards, k=1), (W_f-cx_f-1, cy_f))
                kernels_backward[(H-x-1, y)] = (np.rot90(k_backwards, k=1), (W_b-cx_b-1, cy_b))

                # 4th quadrant
                kernels_forward[(H-y-1, W-x-1)] = (np.rot90(k_forwards, k=2), (H_f-cy_f-1, W_f-cx_f-1))
                kernels_backward[(H-y-1, W-x-1)] = (np.rot90(k_backwards, k=2), (H_b-cy_b-1, W_b-cx_b-1))

                # 1st quadrant
                kernels_forward[(x, W-y-1)] = (np.rot90(k_forwards, k=3), (cx_f, H_f-cy_f-1))
                kernels_backward[(x, W-y-1)] = (np.rot90(k_backwards, k=3), (cx_b, H_b-cy_b-1))

        self.z_kernels_forwards = kernels_forward
        self.z_kernels_backwards = kernels_backward
        self.max_dim = max_dim

        return kernels_forward, kernels_backward, max_dim
    
    def calculate_kernel_coordinates(self, theta, r, center_x, center_y, x, y, kernel_center):
        global_x = r * np.cos(theta)
        global_y = r * np.sin(theta)
        image_x = global_x + center_x
        image_y = center_y - global_y
        kernel_x = image_x - (x - kernel_center)
        kernel_y = image_y - (y - kernel_center)
        ui = int(round(kernel_x))
        vi = int(round(kernel_y))
        return ui, vi
    
    def determine_kernel_bounds(self, x, y, max_x, min_x, max_y, min_y):
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y
        return max_x, min_x, max_y, min_y
    
    def blur_with_angular_velocities_coupled(self, img_f, wx, wy, wz, T, f, pixel_size,
                                            std=0.0, p=0.0, num_samples=200, min_kernel_size=5):
        self.wx, self.wy, self.wz, self.T = wx, wy, wz, T

        self.z_kernels_forwards, self.z_kernels_backwards, self.max_dim = self.h_xyz_kernel_coupled_centered(wx, wy, wz, T, img_f.shape, f, pixel_size)
        self.xy_kernel = np.array([1])
        self.f = f
        self.pixel_size = pixel_size

        blurred = ImageOperator.scatter(img_f, self.z_kernels_forwards, self.max_dim)
        blurred = ImageOperator.gaussian_noise(blurred, std)
        blurred = ImageOperator.salt_pepper_noise(blurred, p)
        return blurred

    def h_xyz_kernel_coupled_centered(self, wx, wy, wz, T, image_shape, f, pixel_size,
                                  num_samples=200, min_kernel_size=5):
        """
        Reference: Motion Blurred Star Image Restoration Based on MEMS Gyroscope Aid and Blur Kernel Correction
        """
        H, W = image_shape
        center_y_image = H // 2
        center_x_image = W // 2
        fp = float(f) / float(pixel_size)
        t_values = np.linspace(-0.5 * T, 0.5 * T, num_samples)
        dt = float(t_values[1] - t_values[0]) if num_samples > 1 else float(T)
        kernels_forward = {}
        kernels_backward = {}
        max_dim = min_kernel_size

        for y0 in range(H):
            for x0 in range(W):
                x_mid = float(x0 - center_x_image)
                y_mid = float(y0 - center_y_image)

                if abs(wx) < 1e-12 and abs(wy) < 1e-12 and abs(wz) < 1e-12:
                    kernel = np.zeros((1, 1), np.float32)
                    kernel[0, 0] = 1.0
                    kernels_forward[(y0, x0)] = (kernel, (0, 0))
                    kernels_backward[(y0, x0)] = (kernel, (0, 0))
                    continue

                points = []
                x, y = x_mid, y_mid
                img_x = x + center_x_image
                img_y = y + center_y_image 
                points.append((img_y, img_x))
                n_half = (num_samples - 1) // 2

                for _ in range(n_half):
                    x, y = self.step_backward(fp, wx, wy, wz, x, y, dt)
                    img_x = x + center_x_image
                    img_y = center_y_image + y
                    points.append((img_y, img_x))

                x, y = x_mid, y_mid
                for _ in range(n_half):
                    x, y = self.step_forward(fp, wx, wy, wz, x, y, dt)
                    img_x = x + center_x_image
                    img_y = center_y_image + y
                    points.append((img_y, img_x))

                kernel_forwards, center, dim = self.make_kernel_from_points(y0, x0, points, min_kernel_size)
                kernel_backwards = np.flip(kernel_forwards, axis=(0, 1))
                kernels_forward[(y0, x0)] = (kernel_forwards, center)
                kernels_backward[(y0, x0)] = (kernel_backwards, center)
                max_dim = max(max_dim, dim)
        return kernels_forward, kernels_backward, max_dim
    

    def step_forward(self, fp, wx, wy, wz, x, y, delta_t):
        x2 = x + y * wz * delta_t + fp * wy * delta_t
        y2 = y - x * wz * delta_t - fp * wx * delta_t
        return x2, y2


    def step_backward(self, fp, wx, wy, wz, x, y, delta_t):
        x2 = x - y * wz * delta_t - fp * wy * delta_t
        y2 = y + x * wz * delta_t + fp * wx * delta_t
        return x2, y2


    def make_kernel_from_points(self, y0, x0, points, min_kernel_size):
        points = np.asarray(points, dtype=np.float32)
        dy = points[:, 0] - y0
        dx = points[:, 1] - x0
        min_dy = int(np.floor(dy.min()))
        max_dy = int(np.ceil(dy.max()))
        min_dx = int(np.floor(dx.min()))
        max_dx = int(np.ceil(dx.max()))

        # Create empty kernel
        kernel_height = max(max_dy - min_dy + 1, min_kernel_size)
        kernel_width = max(max_dx - min_dx + 1, min_kernel_size)
        kernel = np.zeros((kernel_height, kernel_width), np.float32)
        center_y = -min_dy
        center_x = -min_dx

        # Convert into pixels
        for dyy, dxx in zip(dy, dx):
            iy = int(np.round(dyy)) + center_y
            ix = int(np.round(dxx)) + center_x
            if 0 <= iy < kernel_height and 0 <= ix < kernel_width:
                kernel[iy, ix] += 1.0

        # Normalize
        total = kernel.sum()
        if total > 0:
            kernel /= total
        else:
            kernel[:] = 0
            kernel[center_y, center_x] = 1.0
        return kernel, (center_y, center_x), max(kernel_height, kernel_width)
