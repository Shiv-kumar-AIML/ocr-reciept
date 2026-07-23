import cv2
import numpy as np
from PIL import Image


def apply_clahe_enhancement(img_bgr: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the LAB color space L-channel.
    Enhances contrast under uneven lighting, shadows, glare, and complex background surfaces.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 corner points deterministically into:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest difference

    return rect


def four_point_transform(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies 4-point perspective warp transformation to produce a flat, unwarped top-down rectangular crop.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 20 or maxHeight <= 20:
        return img_bgr

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
    return warped


def detect_document_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Locates 4-corner document quadrilateral polygon contour using bilateral filtering,
    multi-scale Otsu/Canny edge maps, and convex hull analysis.
    Returns 4 corner points array if found (>= 10% of image area), else None.
    """
    h, w = img_bgr.shape[:2]
    total_area = w * h

    # Bilateral filter preserves sharp document edges while smoothing background textures
    filtered = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Multi-scale edge map
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    high_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.4 * high_thresh
    edged = cv2.Canny(blur, low_thresh, high_thresh)

    # Morphological dilation to connect paper edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.10 * total_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)

    return None


def crop_content_density(img_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback cropper using adaptive text/content density bounds with padding.
    """
    h, w = img_bgr.shape[:2]
    total_area = w * h
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    dilated_text = cv2.dilate(adaptive, kernel_text, iterations=2)

    contours, _ = cv2.findContours(dilated_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    for c in contours:
        tx, ty, tw, th = cv2.boundingRect(c)
        if (tx <= 2 or ty <= 2 or tx + tw >= w - 2 or ty + th >= h - 2) and tw * th < 0.05 * total_area:
            continue
        if cv2.contourArea(c) > 50:
            valid_boxes.append((tx, ty, tx + tw, ty + th))

    if valid_boxes:
        pad_x = int(w * 0.03)
        pad_y = int(h * 0.02)
        min_x = max(0, min(b[0] for b in valid_boxes) - pad_x)
        min_y = max(0, min(b[1] for b in valid_boxes) - pad_y)
        max_x = min(w, max(b[2] for b in valid_boxes) + pad_x)
        max_y = min(h, max(b[3] for b in valid_boxes) + pad_y)

        crop_w = max_x - min_x
        crop_h = max_y - min_y
        if (crop_w * crop_h) < 0.96 * total_area and crop_w > 50 and crop_h > 50:
            return img_bgr[min_y:max_y, min_x:max_x]

    return img_bgr


def scan_and_crop_document(pil_img: Image.Image, enhance_contrast: bool = True) -> Image.Image:
    """
    Main Production Document Scanner Entry Point:
    1. Optionally applies LAB CLAHE contrast enhancement.
    2. Detects 4-corner document quadrilateral polygon.
    3. Performs 4-point perspective warp transformation if polygon is found.
    4. Falls back to content density bounding crop if 4-point quad is obscured.
    Returns processed PIL Image.
    """
    img_np = np.array(pil_img)
    if img_np is None or img_np.size == 0:
        return pil_img

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 and img_np.shape[2] == 3 else img_np

    # Apply CLAHE contrast enhancement
    processed_bgr = apply_clahe_enhancement(img_bgr) if enhance_contrast else img_bgr

    # Try 4-point perspective transform
    quad = detect_document_quad(processed_bgr)
    if quad is not None:
        scanned_bgr = four_point_transform(processed_bgr, quad)
    else:
        scanned_bgr = crop_content_density(processed_bgr)

    scanned_rgb = cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(scanned_rgb)
