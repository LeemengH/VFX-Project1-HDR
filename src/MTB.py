import cv2
import numpy as np

def median_threshold_bitmap(image):
    """
    將圖像轉換為 Median Threshold Bitmap (MTB)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = np.median(gray)
    bitmap = np.where(gray >= median, 1, 0).astype(np.uint8)
    return bitmap

def shift_image(image, dx, dy):
    """
    根據給定的 dx, dy 來移動圖像
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)

def align_images(images):
    """
    對齊多張圖片，假設第一張圖片為基準。
    """
    base_image = images[0]
    base_mtb = median_threshold_bitmap(base_image)
    aligned_images = [base_image]
    
    for img in images[1:]:
        best_dx, best_dy = 0, 0
        min_diff = float('inf')
        mtb = median_threshold_bitmap(img)
        
        for dx in range(-10, 11):  # 搜索 -10 到 10 像素範圍內的最佳對齊
            for dy in range(-10, 11):
                shifted_mtb = shift_image(mtb, dx, dy)
                diff = np.sum(np.abs(shifted_mtb - base_mtb))
                
                if diff < min_diff:
                    min_diff = diff
                    best_dx, best_dy = dx, dy
        
        aligned_img = shift_image(img, best_dx, best_dy)
        aligned_images.append(aligned_img)
    
    return aligned_images

# 測試
if __name__ == "__main__":
    image_files = ["aligned_0.jpg", "aligned_1.jpg", "aligned_2.jpg", "aligned_3.jpg", "aligned_4.jpg"]  # 載入圖片列表
    images = [cv2.imread(f) for f in image_files]
    aligned_images = align_images(images)
    
    for i, img in enumerate(aligned_images):
        cv2.imwrite(f"aligned_{i}.jpg", img)  # 儲存對齊後的圖片