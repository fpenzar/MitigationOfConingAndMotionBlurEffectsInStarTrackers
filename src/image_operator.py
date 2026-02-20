import numpy as np

class ImageOperator:

    def __init__(self):
        pass

    @classmethod
    def correlate(cls, image, kernel):
        image = np.asarray(image)
        kernel = np.asarray(kernel)
        kernel_height, kernel_width = kernel.shape
        pad_y = kernel_height // 2
        pad_x = kernel_width // 2
        if image.ndim != 2:
            raise ValueError(f"Unsupported image ndim={image.ndim}")
        H, W = image.shape
        img_padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)),
                            mode="constant", constant_values=0)
        out = np.zeros_like(image, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                patch = img_padded[y:y+kernel_height, x:x+kernel_width]
                out[y, x] = np.sum(patch * kernel)
        return out
    
    @classmethod
    def scatter(cls, image, kernels, max_dim):
        image = np.asarray(image)
        pad_y = max_dim // 2
        pad_x = max_dim // 2
        if image.ndim != 2:
            raise ValueError(f"Unsupported image ndim={image.ndim}")
        H, W = image.shape
        img_padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)),
                            mode="constant", constant_values=0)
        out = np.zeros_like(image, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                kernel, kernel_center = kernels[(y, x)]
                kernel_center_y, kernel_center_x = kernel_center
                kernel = np.asarray(kernel)
                kernel_height, kernel_width = kernel.shape
                center_y = y + pad_y
                center_x = x + pad_x
                start_y = center_y - kernel_center_y
                end_y = center_y + (kernel_height - kernel_center_y)
                start_x = center_x - kernel_center_x
                end_x = center_x + (kernel_width - kernel_center_x)
                patch = img_padded[start_y:end_y, start_x:end_x]
                out[y, x] = np.sum(patch * kernel)
        return out
    
    @classmethod
    def convolve(cls, image, kernel):
        """
        2D convolution implemented as correlation with flipped kernel.
        """
        k_flipped = np.flip(kernel, axis=(0, 1))
        return cls.correlate(image, k_flipped)
    
    @classmethod
    def gaussian_noise(cls, image, std):
        noisy = image.copy()
        if std > 0:
            noise = np.random.normal(loc=0.0, scale=std, size=image.shape)
            noisy = noisy + noise
        noisy = np.clip(noisy, 0.0, 1.0)
        return noisy
    
    @classmethod
    def salt_pepper_noise(cls, image, p):
        noisy = image.copy()
        if p <= 0:
            return noisy
        rnd = np.random.rand(*image.shape)
        noisy[rnd < (p / 2)] = 0.0
        noisy[(rnd >= (p / 2)) & (rnd < p)] = 1.0
        return noisy