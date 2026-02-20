import cv2
import numpy as np

class Polar:

    def __init__(self):
        pass

    @classmethod
    def to_polar(cls, img, center=None, out_shape=None, R=None):
        H, W = img.shape
        if center is None:
            center = (W / 2.0, H / 2.0)
        if R is None:
            R = np.hypot(max(center[0], W-center[0]), max(center[1], H-center[1]))
        if out_shape is None:
            # (rows=theta, cols=r)
            out_shape = (int(np.ceil(R)), 4*W) 
        polar = cv2.warpPolar(
            img.astype(np.float32),
            dsize=(out_shape[0], out_shape[1]),  # (theta, r)
            center=center,
            maxRadius=R,
            flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_CUBIC
        )

        # polar = cls.reset_invalid_region(polar, center, H, W, R)

        # polar has shape (rows=theta, cols=r)
        return polar, center, R
    
    @classmethod
    def reset_invalid_region(cls, polar, center, H, W, R):
        # Make the pixels outside the valid range black
        theta_len, r_len = polar.shape
        theta = np.linspace(0, 2*np.pi, theta_len, endpoint=False)
        r = np.linspace(0, R, r_len)
        Theta, Rr = np.meshgrid(theta, r, indexing="ij")
        X = center[0] + Rr * np.cos(Theta)
        Y = center[1] - Rr * np.sin(Theta)
        valid = (
            (X >= 0) & (X < W) &
            (Y >= 0) & (Y < H)
        )
        polar[~valid] = 0.0
        return polar
    
    @classmethod
    def from_polar(cls, polar, H, W, center, R):
        cart = cv2.warpPolar(
            polar.astype(np.float32),
            dsize=(W, H),
            center=center,
            maxRadius=R,
            flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP
        )
        return cart