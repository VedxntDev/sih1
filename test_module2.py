#!/usr/bin/env python3
"""
Module 2: Retinal Structure & Lesion Segmentation Engine
Performs segmentation of:
1. Optic Disc (OD) & Fovea / Macula Center
2. Retinal Blood Vessel Tree (Directional Matched Line Filtering)
3. Microaneurysms (MAs) via Local Green Contrast & Red-Green Differential
4. Hard Lipid Exudates (Local Contrast & Dual-Channel Thresholding)
5. Retinal Blot & Flame Hemorrhages
6. Neovascularization (NV) Loops on Disc Margin (NVD/NVE)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def segment_retinal_structures(enhanced_bgr):
    """
    Segments anatomical structures and clinical lesions from fundus image.
    Returns:
        overlay (ndarray): Color-coded visual overlay (BGR)
        lesion_stats (dict): Quantified clinical telemetry
        masks (dict): Binary masks for each structure/lesion
    """
    img = enhanced_bgr.copy()
    h, w, c = img.shape
    b_chan, g_chan, r_chan = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Circular Field of View (FOV) Mask & Safe Inner Area
    _, fov_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    fov_area = float(np.sum(fov_mask > 0))
    # Erode by 21px to eliminate outer peripheral rim artifacts
    fov_inner = cv2.erode(fov_mask, np.ones((21, 21), np.uint8))

    # 2. Optic Disc Localization & Segmentation (Brightest region in Red + Green inside FOV)
    rg_composite = cv2.GaussianBlur(r_chan.astype(float) * 0.6 + g_chan.astype(float) * 0.4, (31, 31), 0)
    rg_composite[fov_inner == 0] = 0
    _, _, _, max_loc = cv2.minMaxLoc(rg_composite)
    od_x, od_y = max_loc
    od_radius = int(min(h, w) * 0.11)
    
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, (od_x, od_y), int(od_radius * 1.45), 255, -1)

    # 3. Fovea Localization (~3.8 disc diameters temporal to OD)
    if od_x < w // 2:
        fovea_x = min(w - 40, int(od_x + 3.8 * od_radius))
    else:
        fovea_x = max(40, int(od_x - 3.8 * od_radius))
    fovea_y = min(h - 40, max(40, int(od_y + 0.15 * od_radius)))

    fovea_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fovea_mask, (fovea_x, fovea_y), int(1.2 * od_radius), 255, -1)

    # 4. Blood Vessel Tree Segmentation (Directional Line Morphology)
    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    g_closed = cv2.morphologyEx(g_chan, cv2.MORPH_CLOSE, kernel_disk)
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel_disk)
    
    g_diff = g_closed.astype(float) - g_chan.astype(float)
    r_diff = r_closed.astype(float) - r_chan.astype(float)

    vessel_resp = np.zeros((h, w), dtype=np.float32)
    for angle in range(0, 180, 15):
        rad = np.deg2rad(angle)
        dx, dy = int(round(7 * np.cos(rad))), int(round(7 * np.sin(rad)))
        line_k = np.zeros((17, 17), dtype=np.uint8)
        cv2.line(line_k, (8 - dx, 8 - dy), (8 + dx, 8 + dy), 1, 1)
        opened = cv2.morphologyEx(g_diff.astype(np.uint8), cv2.MORPH_OPEN, line_k)
        vessel_resp = np.maximum(vessel_resp, opened.astype(np.float32))

    v_thresh = max(18.0, float(np.percentile(vessel_resp[fov_inner > 0], 91)) if np.any(fov_inner > 0) else 20.0)
    v_cand = ((vessel_resp > v_thresh) & (fov_inner > 0) & (od_mask == 0)).astype(np.uint8) * 255
    num_v, v_labels, v_stats, v_cents = cv2.connectedComponentsWithStats(v_cand)

    vessel_mask = np.zeros((h, w), dtype=np.uint8)
    hem_mask = np.zeros((h, w), dtype=np.uint8)
    hem_count = 0

    for i in range(1, num_v):
        area = v_stats[i, cv2.CC_STAT_AREA]
        bw = v_stats[i, cv2.CC_STAT_WIDTH]
        bh = v_stats[i, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(bw**2 + bh**2)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        comp_mask = (v_labels == i)
        mean_r_diff = np.mean(r_diff[comp_mask])
        mean_r_val = np.mean(r_chan[comp_mask])

        # True Blot Hemorrhage (isolated thick dark blob with area >= 95 and aspect < 2.2, not fovea)
        if 95 <= area <= 700 and diag < 38 and aspect < 2.2 and mean_r_diff > 30 and mean_r_val < 145 and fovea_mask[int(v_cents[i][1]), int(v_cents[i][0])] == 0:
            hem_mask[comp_mask] = 255
            hem_count += 1
        elif area >= 40 or diag >= 25 or aspect >= 2.0:
            vessel_mask[comp_mask] = 255

    vessel_dilated = cv2.dilate(vessel_mask, np.ones((7, 7), np.uint8))
    vessel_area = float(np.sum(vessel_mask > 0))
    vessel_density = vessel_area / (fov_area + 1e-5)

    valid_retina = (fov_inner > 0) & (od_mask == 0) & (vessel_dilated == 0)

    # 5. Hard Exudates: Bright waxy lipid deposits (dual-channel elevated reflectance)
    g_opened = cv2.morphologyEx(g_chan, cv2.MORPH_OPEN, kernel_disk)
    r_opened = cv2.morphologyEx(r_chan, cv2.MORPH_OPEN, kernel_disk)
    g_bright = g_chan.astype(float) - g_opened.astype(float)
    r_bright = r_chan.astype(float) - r_opened.astype(float)

    ex_cand = ((g_bright > 26) & (r_bright > 18) & (g_chan > 135) & (r_chan > 168) & valid_retina).astype(np.uint8) * 255
    num_ex, ex_labels, ex_stats, _ = cv2.connectedComponentsWithStats(ex_cand)
    ex_count = 0
    ex_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_ex):
        if 5 <= ex_stats[i, cv2.CC_STAT_AREA] <= 1500:
            ex_count += 1
            ex_mask[ex_labels == i] = 255
    exudate_area = float(np.sum(ex_mask > 0))

    # 6. Microaneurysms (MAs): Small dark red spots (size 3-45 px, high red-green contrast)
    ma_cand = ((g_diff > 32) & ((r_chan.astype(float) - g_chan.astype(float)) > 48) & valid_retina & (fovea_mask == 0) & (hem_mask == 0)).astype(np.uint8) * 255
    num_ma, ma_labels, ma_stats, _ = cv2.connectedComponentsWithStats(ma_cand)
    ma_count = 0
    ma_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_ma):
        area = ma_stats[i, cv2.CC_STAT_AREA]
        if 3 <= area <= 45:
            ma_count += 1
            ma_mask[ma_labels == i] = 255
    ma_area = float(np.sum(ma_mask > 0))

    # 7. Neovascularization (NV) Detection on Optic Disc Margin (NVD Loops)
    y_g, x_g = np.ogrid[:h, :w]
    od_disc_zone = ((x_g - od_x)**2 + (y_g - od_y)**2 <= (1.1 * od_radius)**2)
    od_dark_structures = (g_chan < 60) & (r_chan > 60) & od_disc_zone
    contours, hier = cv2.findContours(od_dark_structures.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_loop = False
    if hier is not None and len(hier) > 0:
        for c in range(len(hier[0])):
            if hier[0][c][2] != -1 or hier[0][c][3] != -1:
                has_loop = True
                break
    nv_flag = bool(has_loop)

    lesion_stats = {
        'od_center': (int(od_x), int(od_y)),
        'fovea_center': (int(fovea_x), int(fovea_y)),
        'vessel_density': float(vessel_density),
        'ma_count': int(ma_count),
        'ma_area': float(ma_area),
        'exudate_count': int(ex_count),
        'exudate_area': float(exudate_area),
        'hem_count': int(hem_count),
        'hem_area': float(np.sum(hem_mask > 0)),
        'nv_flag': bool(nv_flag),
        'nv_area': float(np.sum(od_dark_structures)) if nv_flag else 0.0
    }

    # Build High-Definition RGB Overlay Image
    overlay = img.copy()
    overlay[vessel_mask > 0] = [0, 255, 0] # Vessels -> Green
    od_perim = cv2.morphologyEx(od_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    overlay[od_perim > 0] = [0, 255, 255] # OD -> Yellow
    fov_perim = cv2.morphologyEx(fovea_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    overlay[fov_perim > 0] = [255, 120, 0] # Fovea -> Blue
    ma_dilated = cv2.dilate(ma_mask, np.ones((3,3), np.uint8))
    overlay[ma_dilated > 0] = [0, 0, 255] # MAs -> Red
    overlay[ex_mask > 0] = [255, 255, 0] # Exudates -> Cyan
    overlay[hem_mask > 0] = [255, 0, 255] # Hemorrhages -> Magenta

    masks = {
        'fov': fov_mask,
        'od': od_mask,
        'fovea': fovea_mask,
        'vessels': vessel_mask,
        'mas': ma_mask,
        'exudates': ex_mask,
        'hemorrhages': hem_mask
    }

    return overlay, lesion_stats, masks

if __name__ == "__main__":
    print("Module 2 structure and lesion segmentation engine compiled successfully.")


