"""
cv2_crop.py
-----------
Fast, lightweight OpenCV-based leaf localizer replacing YOLO.
"""

import cv2
import numpy as np
from PIL import Image


class LeafLocalizer:
    def __init__(self, min_area_ratio: float = 0.05, padding: int = 10):
        self.min_area_ratio = min_area_ratio
        self.padding = padding

    def crop_leaf(self, image_path: str) -> Image.Image:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return Image.open(image_path).convert("RGB")

        h, w = img_bgr.shape[:2]
        total_area = h * w

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        _, a_channel, _ = cv2.split(lab)

        green_mask = cv2.bitwise_not(a_channel)
        green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)

        _, thresh = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) < (self.min_area_ratio * total_area):
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        x, y, cw, ch = cv2.boundingRect(largest_cnt)

        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w, x + cw + self.padding)
        y2 = min(h, y + ch + self.padding)

        cropped = img_bgr[y1:y2, x1:x2]
        
        # Guard against zero-area crops
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))