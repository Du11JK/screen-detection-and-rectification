import cv2
import numpy as np
from quadrilateral_fitter import process_image
from screen_detector import read_image
import os

# Common screen aspect ratios
SCREEN_RATIOS = {
    '16:9': (16, 9),
    '16:10': (16, 10),
    '4:3': (4, 3),
    '21:9': (21, 9),
    '3:2': (3, 2),
    '5:4': (5, 4),
    '18:9': (18, 9),
    '19:9': (19, 9),
    '2:1': (2, 1),
}


def order_points(pts):
    """
    Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: List of four (x, y) coordinates.

    Returns:
        rect: Numpy array of ordered points.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def calculate_best_ratio(quad_points):
    """
    Calculate the most likely screen aspect ratio based on a quadrilateral.

    Args:
        quad_points: Vertices of the quadrilateral.

    Returns:
        best_ratio: Name of the best matching ratio.
        ratio_value: Tuple (width, height) of the ratio.
    """
    ordered_pts = order_points(quad_points)

    top_width = np.linalg.norm(ordered_pts[1] - ordered_pts[0])
    bottom_width = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
    left_height = np.linalg.norm(ordered_pts[3] - ordered_pts[0])
    right_height = np.linalg.norm(ordered_pts[2] - ordered_pts[1])

    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2

    current_ratio = avg_width / avg_height

    best_ratio = None
    min_diff = float('inf')

    for ratio_name, (w, h) in SCREEN_RATIOS.items():
        standard_ratio = w / h
        diff = abs(current_ratio - standard_ratio)
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio_name

    return best_ratio, SCREEN_RATIOS[best_ratio]

def create_corrected_image(image, quad_points, ratio=(16, 9), output_size=(1920, 1080)):
    """
    Apply perspective transformation to correct the screen image.

    Args:
        image: Original image.
        quad_points: Detected quadrilateral vertices.
        ratio: Target aspect ratio (width, height).
        output_size: Output image size (width, height).

    Returns:
        corrected_image: The perspective-corrected image.
        transform_matrix: The transformation matrix.
    """
    src_points = order_points(quad_points)

    width, height = output_size
    if ratio[0] / ratio[1] != width / height:
        # Adjust to maintain the aspect ratio
        if ratio[0] / ratio[1] > width / height:
            height = int(width * ratio[1] / ratio[0])
        else:
            width = int(height * ratio[0] / ratio[1])

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    return corrected_image, transform_matrix

def process_screen_correction(
    image_path,
    model_path='./models/sam2.1_b.pt',
    output_dir='./output',
    auto_ratio=True,
    manual_ratio=None,
    output_size=(1920, 1080)
):
    """
    Complete screen correction pipeline.

    Args:
        image_path: Path to input image.
        model_path: Path to SAM model.
        output_dir: Output directory.
        auto_ratio: Whether to detect aspect ratio automatically.
        manual_ratio: Manually specified ratio name (e.g., '16:9').
        output_size: Output image size.

    Returns:
        results: Dictionary containing all results.
    """
    os.makedirs(output_dir, exist_ok=True)

    quad_points = process_image(image_path, model_path, None)
    if quad_points is None:
        return None

    image = read_image(image_path)

    if auto_ratio:
        best_ratio_name, ratio_value = calculate_best_ratio(quad_points)
    else:
        if manual_ratio and manual_ratio in SCREEN_RATIOS:
            ratio_value = SCREEN_RATIOS[manual_ratio]
            best_ratio_name = manual_ratio
        else:
            ratio_value = SCREEN_RATIOS['16:9']
            best_ratio_name = '16:9'

    results = {
        'detected_quad': quad_points,
        'best_ratio': best_ratio_name,
        'corrected_images': {}
    }

    # Only generate correction for the best ratio
    corrected_img, transform_matrix = create_corrected_image(
        image, quad_points, ratio_value, output_size
    )

    output_filename = f'corrected_{best_ratio_name.replace(":", "_")}.jpg'
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, corrected_img)

    results['corrected_images'][best_ratio_name] = {
        'path': output_path,
        'size': corrected_img.shape[:2],
        'transform_matrix': transform_matrix
    }

    return results


if __name__ == '__main__':
    # Example usage for standalone testing
    image_path = './images/IMG_2010.HEIC'
    model_path = './models/sam2.1_b.pt'
    output_dir = './output'

    results = process_screen_correction(
        image_path=image_path,
        model_path=model_path,
        output_dir=output_dir,
        auto_ratio=True,
        output_size=(1920, 1080)
    )