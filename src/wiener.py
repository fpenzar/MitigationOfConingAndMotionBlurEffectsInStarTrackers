import numpy as np
from degradation import Degradation
from image_operator import ImageOperator
from polar import Polar
import math
import cv2


class Wiener:

    def __init__(self, degradation: Degradation, K=1e-2):
        self.degradation = degradation
        self.theta_rad = degradation.wz * degradation.T
        if K == 0:
            self.K = 1e-2
        else:
            self.K = K
    
    def restore(self, image):
        H, W = image.shape
        polar_blurred, center, R = Polar.to_polar(image)
        polar_z_deblurred = self.theta_deconv_polar(polar_blurred, self.theta_rad, image.shape[1], centered=True)
        polar_z_deblurred = Polar.reset_invalid_region(polar_z_deblurred, center, H, W, R)
        z_deblurred = Polar.from_polar(polar_z_deblurred, *image.shape, center, R)
        xyz_deblurred = self.xy_deconv(z_deblurred)
        return xyz_deblurred, polar_z_deblurred, z_deblurred
    
    def xy_deconv(self, image):
        blurred = np.asarray(image, dtype=np.float32)
        psf = np.asarray(self.degradation.xy_kernel, dtype=np.float32)
        H, W = blurred.shape
        ph, pw = psf.shape
        center_y, center_x = ph//2, pw//2
        y0 = H//2 - center_y
        x0 = W//2 - center_x

        psf_pad = np.zeros((H, W), dtype=np.float32)
        psf_pad[y0:y0+ph, x0:x0+pw] = psf
        psf_pad = np.fft.ifftshift(psf_pad)

        G = np.fft.fft2(blurred)
        Hf = np.fft.fft2(psf_pad)

        Hf_conj = np.conj(Hf)
        denominator = (np.abs(Hf) ** 2) + self.K
        F_hat = (Hf_conj / denominator) * G
        restored = np.fft.ifft2(F_hat).real

        restored = np.clip(restored, 0.0, 1.0)
        return restored
    
    def theta_deconv_polar(self, polar, dtheta_rad, W, centered=True):
        polar = np.asarray(polar, dtype=np.float32)
        Ttheta, Rn = polar.shape
        if dtheta_rad < 1e-3:
            return polar
        L = int(np.round((abs(dtheta_rad) / (2*np.pi)) * Ttheta))
        L = max(L, 1)

        h = np.zeros(Ttheta, dtype=np.float32)
        if centered:
            half = L // 2
            h[:half+1] = 1.0
            h[-half:] = 1.0
        else:
            h[:L] = 1.0
        h /= h.sum()

        Hf = np.fft.fft(h)
        Hf_conj = np.conj(Hf)
        denominator = (np.abs(Hf) ** 2) + self.K

        out = np.empty_like(polar, dtype=np.float32)
        # Apply the 1D deconv for each r
        for r in range(Rn):
            G = np.fft.fft(polar[:, r])
            F_hat = (Hf_conj / denominator) * G
            out[:, r] = np.fft.ifft(F_hat).real

        return np.clip(out, 0.0, 1.0)
    
    def coupled_rotation_center_pixels(self, wx, wy, wz, f, pixel_size, cx, cy, eps=1e-9):
        if abs(wz) < eps:
            return None

        fp = float(f) / float(pixel_size)
        x_c = -(fp * wx) / wz
        y_c = -(fp * wy) / wz
        u_c = cx + x_c
        v_c = cy + y_c
        return (float(u_c), float(v_c))
    
    def restore_coupled(self, image):
        H, W = image.shape
        cx, cy = (W / 2.0, H / 2.0)
        center = self.coupled_rotation_center_pixels(
            wx=self.degradation.wx,
            wy=self.degradation.wy,
            wz=self.degradation.wz,
            f=self.degradation.f,
            pixel_size=self.degradation.pixel_size,
            cx=cx,
            cy=cy
        )
        # If wz = 0, polar deconv not applicable
        if center is None or abs(self.theta_rad) < 1e-3:
            return image, None, image
        polar_blurred, center_used, R = Polar.to_polar(image, center=center)
        polar_deblurred = self.theta_deconv_polar(polar_blurred, self.theta_rad, W, centered=True)
        # polar_blurred = Polar.reset_invalid_region(polar_deblurred, center_used, H, W, R)
        restored = Polar.from_polar(polar_deblurred, H, W, center_used, R)

        return np.clip(restored, 0.0, 1.0), polar_deblurred, restored