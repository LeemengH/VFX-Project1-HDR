import cv2
import numpy as np
import glob
import os
import re

def compute_mtb(img):
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    median_val = np.median(img_gray)
    mtb = (img_gray >= median_val).astype(np.uint8)
    return mtb

def compute_error(mtb_ref, mtb_img, offset):
    shifted = np.roll(mtb_img, shift=(offset[1], offset[0]), axis=(0,1))
    diff = np.logical_xor(mtb_ref, shifted)
    return np.sum(diff)

def m_tb_align(ref_img, align_img, max_levels=4):
    offset = (0, 0)
    for level in reversed(range(max_levels)):
        scale = 2 ** level
        height, width = ref_img.shape[:2]
        new_size = (width // scale, height // scale)
        ref_scaled = cv2.resize(ref_img, new_size, interpolation=cv2.INTER_LINEAR)
        align_scaled = cv2.resize(align_img, new_size, interpolation=cv2.INTER_LINEAR)

        mtb_ref = compute_mtb(ref_scaled)
        mtb_align = compute_mtb(align_scaled)

        best_offset = offset
        best_error = compute_error(mtb_ref, mtb_align, offset)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                candidate_offset = (offset[0]*2 + dx, offset[1]*2 + dy)
                error = compute_error(mtb_ref, mtb_align, candidate_offset)
                if error < best_error:
                    best_error = error
                    best_offset = candidate_offset
        offset = best_offset
        print(f"Level {level}: Best offset = {offset}, Error = {best_error}")
    
    return offset

def align_images(img_list, max_levels=4):
    if not img_list:
        return []
    ref_img = img_list[0]
    offsets = [(0, 0)]
    for idx, align_img in enumerate(img_list[1:], start=1):
        print(f"Aligning image {idx} to reference image...")
        offset = m_tb_align(ref_img, align_img, max_levels)
        offsets.append(offset)
    return offsets

def natural_key(string):
    """自然排序鍵（可以按照檔名中數字排序）"""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def align_images_folder(input_folder, output_folder, max_levels=4):
    """
    讀取 input_folder 裡面所有 .jpg 圖片（根據檔名數字自然排序，然後反轉），
    按順序對齊，並依序輸出 aligned_0.jpg, aligned_1.jpg ...
    保證 aligned_0 對應曝光度最大圖。
    """
    img_paths = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_key, reverse=False)
    if not img_paths:
        print(f"No JPG images found in {input_folder}.")
        return
    
    print("Image loading order (largest exposure first):")
    for p in img_paths:
        print(p)
    
    img_list = [cv2.imread(path) for path in img_paths]
    offsets = align_images(img_list, max_levels)

    os.makedirs(output_folder, exist_ok=True)
    for i, (img, off, src_name) in enumerate(zip(img_list, offsets, img_paths)):
        aligned_img = np.roll(img, shift=(off[1], off[0]), axis=(0, 1))
        output_path = os.path.join(output_folder, f"aligned_{i}.jpg")
        cv2.imwrite(output_path, aligned_img)
        print(f"Saved aligned image {i} (from {os.path.basename(src_name)}) to {output_path}")

if __name__ == "__main__":
    # 範例：如果要測試
    # align_images_folder("your_input_folder", "your_output_folder")
    pass
