import cv2
import numpy as np
import os
from screen_detector import get_screen_mask_api, read_image

def fit_quadrilateral(mask):
    """
    Fit the best quadrilateral to a binary mask.

    Args:
        mask: Binary mask (H, W).

    Returns:
        quad_points: List of four (x, y) vertices of the quadrilateral.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate contour to a quadrilateral using Douglas-Peucker algorithm
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If not a quadrilateral, use minimum area rectangle
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.int32(box).reshape(-1, 1, 2)

    quad_points = [(int(point[0][0]), int(point[0][1])) for point in approx]
    return quad_points

def visualize_result(image_path, mask, quad_points, output_path=None):
    """
    Visualize the result: original image, mask, and quadrilateral overlay.

    Args:
        image_path: Path to the original image.
        mask: Detected mask.
        quad_points: Quadrilateral vertices.
        output_path: Path to save the visualization.
    """
    image = read_image(image_path)
    h, w = image.shape[:2]

    # Resize for visualization
    target_h = 400
    scale = target_h / h
    target_w = int(w * scale)

    img_resized = cv2.resize(image, (target_w, target_h))
    mask_3ch = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_3ch, (target_w, target_h))

    result_image = img_resized.copy()
    if quad_points:
        # Scale quadrilateral points
        scaled_points = [(int(p[0] * scale), int(p[1] * scale)) for p in quad_points]
        pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(result_image, [pts], True, (0, 0, 255), 2)
        for i, point in enumerate(scaled_points):
            cv2.circle(result_image, point, 5, (0, 255, 0), -1)
            cv2.putText(result_image, str(i+1), (point[0]+8, point[1]+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    gap = np.ones((target_h, 10, 3), dtype=np.uint8) * 255
    combined = np.hstack([img_resized, gap, mask_resized, gap, result_image])

    # Add titles
    title_height = 30
    title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_img, 'Original', (target_w//2-30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(title_img, 'Mask', (target_w + 10 + target_w//2-15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(title_img, 'Quadrilateral', (2*target_w + 20 + target_w//2-40, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    final_result = np.vstack([title_img, combined])

    if output_path:
        cv2.imwrite(output_path, final_result)

def process_image(image_path, model_path='./models/sam2.1_b.pt', output_path=None):
    """
    Full processing pipeline: read image, get mask, fit quadrilateral, visualize.

    Args:
        image_path: Path to input image.
        model_path: Path to SAM model.
        output_path: Path to save visualization.

    Returns:
        quad_points: Fitted quadrilateral vertices.
    """
    mask = get_screen_mask_api(image_path, model_path)
    if mask is None:
        return None

    quad_points = fit_quadrilateral(mask)
    if quad_points is None:
        return None

    if output_path is not None:
        visualize_result(image_path, mask, quad_points, output_path)

    return quad_points

if __name__ == '__main__':
    # Example usage for standalone testing
    image_path = './images/IMG_2010.HEIC'
    model_path = './models/sam2.1_b.pt'
    output_path = './output/quadrilateral_result.jpg'

    os.makedirs('./output', exist_ok=True)
    quad_points = process_image(image_path, model_path, output_path)