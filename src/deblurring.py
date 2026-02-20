import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

import numpy as np, random
np.random.seed(0)
random.seed(0)

from centroids import Centroids
from degradation import Degradation
from rl import RL
from wiener import Wiener
from polar import Polar
from denoiser import Denoiser

import time
import matplotlib.pyplot as plt


def plot_times():
    methods = ["Wiener", "RL (single iteration)", "Denoising"]
    times_ms = [52.381, 542.646, 611.43]

    plt.figure(figsize=(4, 4))
    bars = plt.bar(methods, times_ms, color=["tab:blue", "tab:orange", "tab:red"])
    plt.ylabel("Time [ms]")
    plt.title("Runtime")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


def crop(img, crop_h, crop_w):
    H, W = img.shape
    start_y = max(0, (H - crop_h) // 2)
    start_x = max(0, (W - crop_w) // 2)
    end_y = start_y + crop_h
    end_x = start_x + crop_w
    return img[start_y:end_y, start_x:end_x]


def test_denoiser(original, noisy):
    denoised = Denoiser.denoise_curvature_energy(noisy, 0.1)
    cv2.imshow("Original", original)
    cv2.imshow("Denoised", denoised)
    cv2.imshow("Noisy", noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im_path = r"../data/test_images/image3.jpg"
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    img_f = img.astype(np.float32) / 255.0
    size = 256
    img_f = crop(img_f, size, size)
    f = 0.02
    pixel_size = 5e-4
    T = 0.25
    threshold = 0.4
    std = 0.05
    # std = 0
    p = 0.01
    # p = 0
    denoiser_T = 0.1

    wx = 1
    wy = 1
    wz = 1

    use_rl = False
    use_rl_automatic = False
    denoise = False
    decouple = False
    couple = True

    cv2.imshow("Original", img_f)

    if decouple:
        degradation = Degradation()
        blurred = degradation.blur_with_angular_velocities(img_f, wx, wy, wz, T, f, pixel_size, std, p)
        if denoise:
            blurred = Denoiser.denoise_curvature_energy(blurred, denoiser_T)
            cv2.imshow("Blurred decoupled denoised", blurred)
        else:
            cv2.imshow("Blurred decoupled", blurred)
        mean_blurred_distance = Centroids.evaluate(img_f, blurred, threshold, threshold / 2, "Blurred image")
        wiener = Wiener(degradation, K=std)
        deblurred_wiener, polar_z_deblurred, z_deblurred = wiener.restore(blurred)
        mean_wiener_distance = Centroids.evaluate(img_f, deblurred_wiener, threshold, threshold, "Wiener")
        cv2.imshow("Wiener decoupled", deblurred_wiener)
        if use_rl:
            rl = RL(degradation, iterations=25)
            deblurred_rl = rl.restore(blurred, coupled=False)
            mean_rl_distance = Centroids.evaluate(img_f, deblurred_rl, threshold, threshold, "RL")
            cv2.imshow("RL decoupled", deblurred_rl)
        if use_rl_automatic:
            deblurred_rl_automatic_termination = rl.restore_with_automatic_termination(blurred)
            mean_rl_distance_automatic = Centroids.evaluate(img_f, deblurred_rl_automatic_termination, threshold, threshold, "RL automatic")
            cv2.imshow("RL decoupled automatic", deblurred_rl_automatic_termination)
    
    if couple:
        degradation_coupled = Degradation()
        blurred_coupled = degradation_coupled.blur_with_angular_velocities_coupled(img_f, wx, wy, wz, T, f, pixel_size, std, p)
        if denoise:
            t0 = time.perf_counter()
            blurred_coupled = Denoiser.denoise_curvature_energy(blurred_coupled, denoiser_T)
            t1 = time.perf_counter()
            print(f"Denoising time: {t1-t0} s")
            cv2.imshow("Blurred coupled denoised", blurred_coupled)
        else:
            cv2.imshow("Blurred coupled", blurred_coupled)
        mean_blurred_distance_coupled = Centroids.evaluate(img_f, blurred_coupled, threshold, threshold, "Blurred image coupled")
        wiener_coupled = Wiener(degradation_coupled, K=std)
        deblurred_wiener_coupled, polar_z_deblurred_coupled, z_deblurred_coupled = wiener_coupled.restore_coupled(blurred_coupled)
        mean_wiener_distance = Centroids.evaluate(img_f, deblurred_wiener_coupled, threshold, threshold, "Wiener coupled")
        cv2.imshow("Wiener coupled", deblurred_wiener_coupled)
        if use_rl:
            rl_coupled = RL(degradation_coupled, iterations=25)
            deblurred_rl_coupled = rl_coupled.restore(blurred_coupled, coupled=True)
            mean_rl_distance_coupled = Centroids.evaluate(img_f, deblurred_rl_coupled, threshold, threshold, "RL Coupled")
            cv2.imshow("RL coupled", deblurred_rl_coupled)
        if use_rl_automatic:
            deblurred_rl_automatic_termination_coupled = rl_coupled.restore_with_automatic_termination(blurred_coupled, coupled=True)
            mean_rl_distance_automatic_coupled = Centroids.evaluate(img_f, deblurred_rl_automatic_termination_coupled, threshold, threshold, "RL automatic coupled")
            cv2.imshow("RL coupled automatic", deblurred_rl_automatic_termination_coupled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
