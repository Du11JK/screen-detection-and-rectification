#!/usr/bin/env python3
"""
简洁演示：屏幕检测、四边形拟合、透视矫正完整流程
"""

from perspective_corrector import process_screen_correction
import os

def main():
    image_path = './images/IMG_2010.HEIC'
    model_path = './models/sam2.1_b.pt'
    output_dir = './output'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # 执行完整流程
    results = process_screen_correction(
        image_path=image_path,
        model_path=model_path,
        output_dir=output_dir,
        auto_ratio=True,
        output_size=(1920, 1080)
    )
    
    if results:
        print(f"Success: Generated corrected image with ratio {results['best_ratio']}")
        print(f"Output: {list(results['corrected_images'].values())[0]['path']}")
    else:
        print("Failed to process image")

if __name__ == '__main__':
    main()