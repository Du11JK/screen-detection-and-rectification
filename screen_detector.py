import os
import cv2
import numpy as np
from ultralytics import SAM
from PIL import Image
import pillow_heif

def read_image(image_path):
    """
    Read an image from the given path, supporting HEIC/HEIF and standard formats.

    Args:
        image_path: Path to the image file.

    Returns:
        img: Image in OpenCV BGR format.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.heic', '.heif']:
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        # Convert PIL image to OpenCV format
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return img

def get_image_center(image_path):
    """
    Get the center coordinates of an image.

    Args:
        image_path: Path to the image file.

    Returns:
        (center_x, center_y): Tuple of center coordinates.
    """
    img = read_image(image_path)
    height, width = img.shape[:2]
    return width // 2, height // 2

def get_screen_mask_api(image_path, model_path='./models/sam2.1_b.pt'):
    """
    API: Get the binary mask of the screen from an image.

    Args:
        image_path: Path to the image.
        model_path: Path to the SAM model.

    Returns:
        mask: Binary mask (H, W) as numpy array, or None if not detected.
    """
    model = SAM(model_path)
    center_x, center_y = get_image_center(image_path)

    results = model(image_path, device='mps', retina_masks=True, imgsz=1024,
                   points=[[center_x, center_y]], labels=[1])

    if results[0].masks is not None:
        masks = results[0].masks
        if len(masks) > 1:
            areas = [mask.sum().item() for mask in masks]
            largest_idx = np.argmax(areas)
            selected_mask = masks[largest_idx]
        else:
            selected_mask = masks[0]

        mask_array = selected_mask.data.cpu().numpy().squeeze()
        return (mask_array > 0.5).astype(np.uint8)

    return None

def detect_screen(image_path, model_path, output_dir, prompt_type='everything', **prompt_args):
    """
    Detect the screen in an image and save the result visualization.

    Args:
        image_path: Path to the image.
        model_path: Path to the SAM model.
        output_dir: Directory to save results.
        prompt_type: Type of prompt for segmentation.
        **prompt_args: Additional arguments for the model.

    Returns:
        Number of detected masks (segments).
    """
    model = SAM(model_path)
    center_x, center_y = get_image_center(image_path)

    results = model(image_path, device='mps', retina_masks=True, imgsz=1024,
                   points=[[center_x, center_y]], labels=[1])

    # If multiple segments detected, select the largest one
    if results[0].masks is not None:
        masks = results[0].masks
        if len(masks) > 1:
            areas = [mask.sum().item() for mask in masks]
            largest_idx = np.argmax(areas)
            results[0].masks = results[0].masks[largest_idx:largest_idx+1]

    # Save result visualization
    output_path = os.path.join(output_dir, f'result_{prompt_type}.jpg')
    results_plotted = results[0].plot()
    cv2.imwrite(output_path, results_plotted)

    # Print detected masks info
    if results[0].masks is not None:
        print(f"Detected {len(results[0].masks)} segments")

    return len(results[0].masks) if results[0].masks is not None else 0

if __name__ == '__main__':
    # Example usage for standalone testing
    model_path = './models/sam2.1_b.pt'
    image_path = './images/IMG_2010.HEIC'
    output_dir = './output'

    os.makedirs(output_dir, exist_ok=True)

    # FastSAM only supports everything segmentation
    num_masks = detect_screen(image_path, model_path, output_dir, prompt_type='everything')

    print(f"Results saved in {output_dir}")
