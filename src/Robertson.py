import numpy as np
import time
import cv2
import argparse
from multiprocessing import Pool

def hat_weighting_function(z, z_min=0, z_max=255):
    """Hat weighting function to give higher weights to mid-range values."""
    mid = 0.5 * (z_min + z_max)
    return np.where(z <= mid, z - z_min, z_max - z)

def estimate_E(g, Z, delta_t, w):
    """Estimate irradiance values E_i given g(Z_ij), exposure times, and weights."""
    Z = Z.astype(int)  # Ensure Z is an integer for indexing
    numerator = np.sum(w * g[Z] * delta_t, axis=1)
    denominator = np.sum(w * delta_t ** 2, axis=1)
    denominator = np.where(denominator == 0, 1e-6, denominator)  # Avoid division by zero
    return numerator / denominator

def estimate_g(E, Z, delta_t):
    """Estimate response function g given E_i, image intensities, and exposure times."""
    Z = Z.astype(int)  # Ensure Z is an integer for indexing
    unique_vals = np.unique(Z)
    g = np.zeros(256)
    
    for m in unique_vals:
        indices = np.where(Z == m)
        if len(indices[0]) > 0 and len(indices[1]) > 0:
            g[m] = np.mean(E[indices[0]] * delta_t[indices[1]].astype(float))
    
    g /= g[128]  # Normalize g(128) = 1
    return g

def robertson_hdr(Z, delta_t, max_iter=10, tol=1e-5):
    """Implementation of Robertson's HDR algorithm."""
    start_time = time.time()
    g = np.linspace(0, 1, 256)  # Initialize g as a linear function
    g[128] = 1  # Enforce normalization
    w = hat_weighting_function(Z)
    prev_loss = float('inf')
    
    for times in range(max_iter):
        print(f"========= {times} / {max_iter} =========")
        E = estimate_E(g, Z, delta_t, w)
        g = estimate_g(E, Z, delta_t)
        
        Z = Z.astype(int)  # Ensure Z is an integer for indexing
        # Compute loss (sum of squared differences)
        loss = np.sum(w * (g[Z] - E[:, None] * delta_t) ** 2)
        if abs(prev_loss - loss) < tol:
            print("Coverage!!!")
            break
        prev_loss = loss
        print(f"========= {times}th iteration finished =========")
        print("--- Total %s seconds ---" % (time.time() - start_time))
    
    return g, E

def process_channel(c, images, delta_t, height, width):
    """
    分別處理每個 color channel：
    將影像 intensities 疊成 Z，執行 robertson_hdr()，回傳 E，再重塑回 height x width。
    """
    start_time = time.time()
    print(f"----- Process {c+1} channel -----")
    Z = np.stack([img[:, :, c] for img in images], axis=2)
    g, E = robertson_hdr(Z.reshape(-1, len(images)), delta_t)
    print(f"--- Total {(time.time() - start_time)} seconds in {c+1} channel---")
    return (c, E.reshape(height, width))

def process_hdr(images, delta_t):
    """Process HDR：從多張影像與曝光時間計算出 HDR image (float32)"""
    height, width, _ = images[0].shape
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)

    # 使用 multiprocessing 平行處理 3 個 channel
    with Pool(processes=3) as pool:
        results = pool.starmap(process_channel, [(c, images, delta_t, height, width) for c in range(3)])

    # 將結果填回 hdr_image
    for c, channel_data in results:
        hdr_image[:, :, c] = channel_data
    
    return hdr_image

def tone_mapping(hdr_image):
    """套用 OpenCV Drago tone mapping 將 HDR 轉成 LDR"""
    tonemap = cv2.createTonemapDrago(gamma=2.2)
    ldr = tonemap.process(hdr_image)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    return ldr

def main():
    """CLI 模式，透過 command line 輸入影像與曝光度，計算 HDR 並儲存結果"""
    parser = argparse.ArgumentParser(description="HDR image processing using Robertson algorithm.")
    parser.add_argument('--images', nargs='+', required=True, help="List of aligned image filenames")
    parser.add_argument('--exposures', nargs='+', required=True, type=float, help="List of exposure times (e.g., 0.0167 0.0333 0.0667)")

    args = parser.parse_args()

    if len(args.images) != len(args.exposures):
        print("Error: The number of images and exposures must match.")
        return

    images = [cv2.imread(img).astype(np.float32) for img in args.images]
    hdr_result = process_hdr(images, np.array(args.exposures, dtype=np.float32))
    cv2.imwrite("output.hdr", hdr_result.astype(np.float32))
    print("HDR image saved as output.hdr")

    ldr_result = tone_mapping(hdr_result)
    cv2.imwrite("output_tonemapped.jpg", ldr_result)
    print("Tone-mapped image saved as output_tonemapped.jpg")

if __name__ == "__main__":
    main()
