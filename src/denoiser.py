import numpy as np


class Denoiser:

    def __init__(self):
        pass
    
    @classmethod
    def denoise_curvature_energy(cls, image, T, alpha = 1.5, iterations = 1):
        img = np.asarray(image).astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        H, W = img.shape
        neighbours = [(-1, -1), (-1, 0), (-1, 1),
                        ( 0, -1),          ( 0, 1),
                        ( 1, -1), ( 1, 0), ( 1, 1)]
        
        for _ in range(iterations):
            padded_img = np.pad(img, ((1, 1), (1, 1)), mode="edge")
            out = img.copy()
            for y in range(H):
                for x in range(W):
                    # Estimation of filtered pixel value
                    fxy = padded_img[y + 1, x + 1]

                    f1 = padded_img[y, x]
                    f2 = padded_img[y, x + 1]
                    f3 = padded_img[y, x + 2]

                    f4 = padded_img[y + 1, x]
                    f5 = padded_img[y + 1, x + 2]
                    
                    f6 = padded_img[y + 2, x]
                    f7 = padded_img[y + 2, x + 1]
                    f8 = padded_img[y + 2, x + 2]

                    d1 = (f4 + f5 + f7) / 3.0 - fxy
                    d2 = (f2 + f3 + f5) / 3.0 - fxy
                    d3 = (f1 + f4 + f2) / 3.0 - fxy
                    d4 = (f7 + f5 + f8) / 3.0 - fxy
                    d5 = (f2 + f7 + f5) / 3.0 - fxy
                    d6 = (f4 + f7 + f6) / 3.0 - fxy
                    d7 = (f2 + f7 + f4) / 3.0 - fxy
                    d8 = (f4 + f2 + f5) / 3.0 - fxy
                    ds = np.array([d1, d2, d3, d4, d5, d6, d7, d8], dtype=np.float32)

                    m = int(np.argmin(np.abs(ds)))
                    f_hat = fxy + ds[m]

                    # Pixel energy computation
                    Es = abs(fxy - f_hat)
                    A = []
                    for (dy, dx) in neighbours:
                        fn = padded_img[y + 1 + dy, x + 1 + dx]
                        A.append(abs(fxy - fn) ** alpha)
                    A = np.sort(np.array(A, dtype=np.float32))
                    Ed = float(np.sum(A[:4]))
                    E = Es + Ed
                    
                    # Noise detection and pixel update
                    out[y, x] = fxy if (E < T) else f_hat
            img = out

        img = np.clip(img, 0.0, 1.0)
        return img
