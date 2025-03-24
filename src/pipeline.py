import argparse
import cv2
import numpy as np
import os
from alignment_MTB import align_images_folder
from tone_mapping import mapping
from Robertson import robertson

def run_alignment(input_folder, output_folder):
    print(f"開始對齊 {input_folder} 下的圖片...")
    align_images_folder(input_folder, output_folder)
    print(f"Alignment 完成，已儲存到 {output_folder}。\n")

def run_hdr(image_paths, exposures):
    print(f"開始 HDR 合成...")
    images = [cv2.imread(img).astype(np.float32) for img in image_paths]
    hdr_result = mapping(images, np.array(exposures, dtype=np.float32))
    output_hdr_path = "output.hdr"
    cv2.imwrite(output_hdr_path, hdr_result.astype(np.float32))
    print(f"HDR 合成完成，已儲存為 {output_hdr_path}\n")
    return output_hdr_path

def run_tonemap(hdr_file, output_file):
    print(f"開始 Tone Mapping...")
    hdr_img = cv2.imread(hdr_file, -1)
    ldr_result = robertson(hdr_img)
    cv2.imwrite(output_file, ldr_result)
    print(f"Tone Mapping 完成，已儲存為 {output_file}\n")

def run_full_pipeline(raw_folder, aligned_folder, exposures):
    # Step1: Alignment
    run_alignment(raw_folder, aligned_folder)
    # Step2: 找出 aligned 資料夾內的圖片
    aligned_images = sorted([os.path.join(aligned_folder, f) for f in os.listdir(aligned_folder) if f.endswith('.jpg')])
    # Step3: HDR 合成
    hdr_file = run_hdr(aligned_images, exposures)
    # Step4: Tone Mapping
    run_tonemap(hdr_file, "full_tonemapped.jpg")

def main():
    parser = argparse.ArgumentParser(description="HDR Pipeline Command Line Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Alignment command
    align_parser = subparsers.add_parser("align", help="對齊圖片")
    align_parser.add_argument("--input_folder", required=True, help="原始圖片資料夾")
    align_parser.add_argument("--output_folder", required=True, help="對齊後圖片輸出資料夾")

    # HDR command
    hdr_parser = subparsers.add_parser("hdr", help="HDR 合成")
    hdr_parser.add_argument("--images", nargs='+', required=True, help="對齊後圖片清單")
    hdr_parser.add_argument("--exposures", nargs='+', type=float, required=True, help="曝光時間清單（float）")

    # Tone mapping command
    tonemap_parser = subparsers.add_parser("tonemap", help="Tone mapping")
    tonemap_parser.add_argument("--input", required=True, help="輸入 HDR 檔案路徑")
    tonemap_parser.add_argument("--output", required=True, help="輸出 tone-mapped 圖片名稱")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="從 Alignment 到 HDR 與 Tone Mapping 一鍵完成")
    full_parser.add_argument("--raw_folder", required=True, help="原始圖片資料夾")
    full_parser.add_argument("--aligned_folder", required=True, help="對齊後圖片輸出資料夾")
    full_parser.add_argument("--exposures", nargs='+', type=float, required=True, help="曝光時間清單（float）")

    args = parser.parse_args()

    if args.command == "align":
        run_alignment(args.input_folder, args.output_folder)
    elif args.command == "hdr":
        run_hdr(args.images, args.exposures)
    elif args.command == "tonemap":
        run_tonemap(args.input, args.output)
    elif args.command == "full":
        run_full_pipeline(args.raw_folder, args.aligned_folder, args.exposures)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
