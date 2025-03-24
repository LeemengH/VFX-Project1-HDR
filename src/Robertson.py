import numpy as np
import time
import cv2
import argparse
from multiprocessing import Pool, Manager

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
        print(f"========= {times}th iteratoion finished =========")
        print("--- Total %s seconds ---" % (time.time() - start_time))
    
    return g, E

def process_channel(c, images, delta_t, height, width, hdr_image):
    start_time = time.time()
    print(f"----- Process {c+1} channel -----")
    Z = np.stack([img[:, :, c] for img in images], axis=2)
    g, E = robertson_hdr(Z.reshape(-1, len(images)), delta_t)
    hdr_image[:, :, c] = E.reshape(height, width)
    print(f"--- Total {(time.time() - start_time)} seconds in {c+1} channel---")
    return (c, E.reshape(height, width))

def process_hdr(images, delta_t):
    """Process three images to compute an HDR image."""
    height, width, _ = images[0].shape
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)
    # 創建共享 hdr_image

    # Using multiprocessing.Pool 
    with Pool(processes=3) as pool:
        results = pool.starmap(process_channel, [(c, images, delta_t, height, width, hdr_image) for c in range(3)])

    # fill the result into hdr_image
    for c, channel_data in results:
        hdr_image[:, :, c] = channel_data
    """
    for c in range(3):  # Process each color channel separately
        start_time = time.time()
        print(f"----- Process {c+1} channel -----")
        Z = np.stack([img[:, :, c] for img in images], axis=2)
        g, E = robertson_hdr(Z.reshape(-1, len(images)), delta_t)
        hdr_image[:, :, c] = E.reshape(height, width)
        print("--- Total %s seconds ---" % (time.time() - start_time))
    """
    return hdr_image

def tone_mapping(hdr_image):
    """Apply tone mapping to convert HDR to LDR."""
    tonemap = cv2.createTonemapDrago(gamma=2.2)
    ldr = tonemap.process(hdr_image)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    return ldr

def robertson():
    parser = argparse.ArgumentParser(description="HDR image processing from given images and exposures.")
    parser.add_argument('--images', nargs='+', required=True, help="List of image filenames (e.g., aligned_0.jpg aligned_2.jpg aligned_3.jpg)")
    parser.add_argument('--exposures', nargs='+', required=True, type=float, help="List of exposure times (e.g., 0.0167 0.0333 0.0667)")

    args = parser.parse_args()

    if len(args.images) != len(args.exposures):
        print("Error: The number of images and exposures must match.")
        return
    
        """Compare with the built-in Library"""
    """
    calibrate = cv2.createCalibrate()
    hdr_opencv = calibrate.process(images, delta_t)
    cv2.imwrite("build_output.hdr", hdr_opencv.astype(np.float32))
    print("Build-in HDR image saved as output.hdr")
    ldr_result = tone_mapping(hdr_result)
    cv2.imwrite("output_tonemapped.jpg", ldr_result)
    print("Tone-mapped image saved as output_tonemapped.jpg")
    =======End========="""
    images = [cv2.imread(img).astype(np.float32) for img in args.images]
    hdr_result = process_hdr(images, np.array(args.exposures, dtype=np.float32))
    cv2.imwrite("output.hdr", hdr_result.astype(np.float32))
    print("HDR image saved as output.hdr")

    ldr_result = tone_mapping(hdr_result)
    cv2.imwrite("output_tonemapped.jpg", ldr_result)
    print("Tone-mapped image saved as output_tonemapped.jpg")


# if __name__ == "__main__":
#     robertson()

"""
# Example usage (simulated data)
np.random.seed(42)
Z = np.random.randint(0, 256, (100, 5))  # 100 pixels, 5 exposures
delta_t = np.array([1, 2, 4, 8, 16])

g, E = robertson_hdr(Z, delta_t)
print("Estimated g:", g)
print("Estimated E:", E)
"""
