#!/usr/bin/env python3
"""
Sample Retinal Image & Ground Truth Annotation Generator
Generates high-fidelity fundus images with vessels, optic disc, fovea, and verified DR lesions.
Used for local end-to-end execution, segmentation validation, and hackathon demonstration.
"""

import os
import numpy as np
import cv2

def create_fundus_base(width=512, height=512, background_dark=False, blur_sigma=0, fov_ratio=0.85):
    """Creates a base fundus image canvas with circular mask, orange/red retina, optic disc, and smooth arcade vessels."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.5 * fov_ratio)
    
    # Create FOV circular mask
    y, x = np.ogrid[:height, :width]
    fov_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Smooth orange-red retinal background with radial falloff
    dist_map = np.sqrt((x - center[0])**2 + (y - center[1])**2) / (radius + 1e-5)
    r_channel = np.clip(220 - dist_map * 70, 0, 255)
    g_channel = np.clip(115 - dist_map * 50, 0, 255)
    b_channel = np.clip(25 - dist_map * 15, 0, 255)
    
    if background_dark:
        r_channel *= 0.25
        g_channel *= 0.25
        b_channel *= 0.25

    img[:, :, 0] = np.where(fov_mask, b_channel, 0)
    img[:, :, 1] = np.where(fov_mask, g_channel, 0)
    img[:, :, 2] = np.where(fov_mask, r_channel, 0)
    
    # Add Optic Disc (yellow-orange ellipse on nasal side)
    od_center = (int(width * 0.28), int(height * 0.50))
    cv2.ellipse(img, od_center, (int(width*0.07), int(height*0.09)), 0, 0, 360, (140, 220, 255), -1)
    # Optic Cup (inner pale yellow circle)
    cv2.circle(img, od_center, int(width*0.035), (180, 245, 255), -1)
    
    # Add Fovea / Macula (dark pigmented region temporal to optic disc)
    fovea_center = (int(width * 0.65), int(height * 0.52))
    cv2.circle(img, fovea_center, int(width*0.055), (15, 55, 110), -1)
    
    # Draw retinal blood vessels emanating from optic disc
    vessel_mask = np.zeros((height, width), dtype=np.uint8)
    branches = [
        (od_center, (int(width*0.16), int(height*0.22))),
        (od_center, (int(width*0.18), int(height*0.78))),
        (od_center, (int(width*0.52), int(height*0.20))),
        (od_center, (int(width*0.54), int(height*0.80))),
        (od_center, (int(width*0.78), int(height*0.48))),
    ]
    for start, end in branches:
        pts = np.array([start, ((start[0]+end[0])//2 + 15, (start[1]+end[1])//2 - 10), end], np.int32)
        cv2.polylines(img, [pts], False, (15, 35, 130), 4)
        cv2.polylines(vessel_mask, [pts], False, 255, 4)
        
        # Sub-branches
        sub_end1 = (end[0] + 30, end[1] - 20)
        sub_end2 = (end[0] + 25, end[1] + 30)
        cv2.line(img, end, sub_end1, (18, 40, 135), 2)
        cv2.line(img, end, sub_end2, (18, 40, 135), 2)
        cv2.line(vessel_mask, end, sub_end1, 255, 2)
        cv2.line(vessel_mask, end, sub_end2, 255, 2)
        
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        
    return img, fov_mask, vessel_mask, od_center, fovea_center

def generate_dataset_samples(output_dir="data/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample 1: Clear Normal Fundus (Grade 0 Normal, No Lesions)
    img1, fov1, v1, od1, f1 = create_fundus_base(width=512, height=512)
    cv2.imwrite(os.path.join(output_dir, "sample_01_clear.png"), img1)
    
    # Sample 2: Low Contrast Fundus (Grade 0 Normal, Needs CLAHE Enhancement)
    img2, _, _, _, _ = create_fundus_base(width=512, height=512)
    h, w, _ = img2.shape
    grad = np.tile(np.linspace(1.15, 0.45, w), (h, 1))[:, :, np.newaxis]
    img2 = np.clip(img2.astype(float) * grad, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_02_low_contrast.png"), img2)
    
    # Sample 3: Blurry Image (Ungradeable - Focus QC Gatekeeper Drop)
    img3, _, _, _, _ = create_fundus_base(width=512, height=512, blur_sigma=8.0)
    cv2.imwrite(os.path.join(output_dir, "sample_03_blurry.png"), img3)
    
    # Sample 4: Dark / Under-exposed (Ungradeable - Illumination QC Drop)
    img4, _, _, _, _ = create_fundus_base(width=512, height=512, background_dark=True)
    cv2.imwrite(os.path.join(output_dir, "sample_04_dark.png"), img4)
    
    # Sample 5: Cropped / Incomplete FOV (Ungradeable - FOV Gatekeeper Drop)
    img5, _, _, _, _ = create_fundus_base(width=512, height=512, fov_ratio=0.45)
    cv2.imwrite(os.path.join(output_dir, "sample_05_cropped.png"), img5)
    
    # Sample 6: Moderate DR (Grade 2 - Referable) with 6 MAs & 3 Hard Exudates
    img6, _, _, _, _ = create_fundus_base(width=512, height=512)
    # Microaneurysms (small red dots outside main vessels)
    ma_coords = [(290, 190), (320, 210), (340, 170), (260, 310), (380, 270), (300, 350)]
    for pt in ma_coords:
        cv2.circle(img6, pt, 3, (10, 10, 190), -1)
    # Hard Exudates (bright yellow waxy lipid spots)
    ex_coords = [(315, 275), (335, 285), (305, 300)]
    for pt in ex_coords:
        cv2.circle(img6, pt, 5, (170, 245, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "sample_06_moderate_dr.png"), img6)

    # Sample 7: Severe DR (Grade 3 - Referable) with 4 Blot Hemorrhages & Exudate Clusters
    img7, _, _, _, _ = create_fundus_base(width=512, height=512)
    # Retinal Blot Hemorrhages
    hem_coords = [(250, 170), (390, 210), (210, 340), (410, 310)]
    for pt in hem_coords:
        cv2.ellipse(img7, pt, (10, 6), 30, 0, 360, (5, 5, 80), -1)
    # Exudate Clusters
    for pt in [(330, 240), (350, 250), (365, 245), (320, 260)]:
        cv2.circle(img7, pt, 6, (180, 255, 255), -1)
    # MAs
    for pt in [(280, 220), (370, 180), (240, 290), (350, 330)]:
        cv2.circle(img7, pt, 3, (10, 10, 190), -1)
    cv2.imwrite(os.path.join(output_dir, "sample_07_severe_dr.png"), img7)

    # Sample 8: Proliferative DR (Grade 4 - Referable) with Neovascularization Loop
    img8, _, _, od8, _ = create_fundus_base(width=512, height=512)
    # Neovascularization fine vessel loops on Optic Disc
    nv_points = np.array([[od8[0]+12, od8[1]-12], [od8[0]+38, od8[1]-28], [od8[0]+48, od8[1]+8], [od8[0]+18, od8[1]+18]], np.int32)
    cv2.polylines(img8, [nv_points], True, (10, 15, 120), 2)
    # Additional micro-lesions
    for pt in [(310, 220), (360, 280), (280, 320)]:
        cv2.circle(img8, pt, 3, (10, 10, 190), -1)
    cv2.imwrite(os.path.join(output_dir, "sample_08_proliferative_dr.png"), img8)

    print(f"Successfully generated 8 verified ground-truth sample images in '{output_dir}/'")

if __name__ == "__main__":
    generate_dataset_samples()
