import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def xray_to_3d(image_path):
    # Load grayscale X-ray
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Depth illusion using blur + emboss
    blurred = gaussian_filter(img, sigma=3)

    depth = img.astype(np.float32) - blurred.astype(np.float32)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to bone-like shading
    bone_3d = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_BONE)

    return bone_3d
