#!/usr/bin/env python3
"""
Sample Retinal Image & Ground Truth Annotation Generator
Generates realistic synthetic fundus images with vessels, optic disc, fovea, and DR lesions.
Used for local end-to-end execution, segmentation validation, and hackathon demonstration.
"""

import os
import numpy as np
import cv2

def create_fundus_base(width=512, height=512, background_dark=False, blur_sigma=0, fov_ratio=0.85):
    """Creates a base fundus image canvas with circular mask, orange/red retina, and optic disc."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.5 * fov_ratio)
    
    # Create FOV circular mask
    y, x = np.ogrid[:height, :width]
    fov_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Orange-red retinal background with radial falloff
    dist_map = np.sqrt((x - center[0])**2 + (y - center[1])**2) / radius
    r_channel = np.clip(220 - dist_map * 80, 0, 255)
    g_channel = np.clip(110 - dist_map * 60, 0, 255)
    b_channel = np.clip(20 - dist_map * 15, 0, 255)
    
    if background_dark:
        r_channel *= 0.25
        g_channel *= 0.25
        b_channel *= 0.25

    img[:, :, 0] = np.where(fov_mask, b_channel, 0)
    img[:, :, 1] = np.where(fov_mask, g_channel, 0)
    img[:, :, 2] = np.where(fov_mask, r_channel, 0)
    
    # Add Optic Disc (bright yellow-white ellipse on nasal side)
    od_center = (int(width * 0.3), int(height * 0.5))
    cv2.ellipse(img, od_center, (int(width*0.07), int(height*0.09)), 0, 0, 360, (180, 240, 255), -1)
    
    # Add Fovea (dark macula area on temporal side ~2.5 disc diameters away)
    fovea_center = (int(width * 0.65), int(height * 0.52))
    cv2.circle(img, fovea_center, int(width*0.06), (10, 40, 90), -1)
    
    # Draw retinal blood vessels emanating from optic disc
    vessel_mask = np.zeros((height, width), dtype=np.uint8)
    branches = [
        (od_center, (int(width*0.15), int(height*0.2))),
        (od_center, (int(width*0.18), int(height*0.8))),
        (od_center, (int(width*0.55), int(height*0.25))),
        (od_center, (int(width*0.58), int(height*0.75))),
        (od_center, (int(width*0.8), int(height*0.5))),
    ]
    for start, end in branches:
        pts = np.array([start, ((start[0]+end[0])//2 + 20, (start[1]+end[1])//2 - 15), end], np.int32)
        cv2.polylines(img, [pts], False, (10, 25, 120), 4)
        cv2.polylines(vessel_mask, [pts], False, 255, 4)
        
        # Sub-branches
        sub_end1 = (end[0] + 30, end[1] - 25)
        sub_end2 = (end[0] + 25, end[1] + 35)
        cv2.line(img, end, sub_end1, (12, 28, 125), 2)
        cv2.line(img, end, sub_end2, (12, 28, 125), 2)
        cv2.line(vessel_mask, end, sub_end1, 255, 2)
        cv2.line(vessel_mask, end, sub_end2, 255, 2)
        
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        
    return img, fov_mask, vessel_mask, od_center, fovea_center

def generate_dataset_samples(output_dir="data/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample 1: Clear Gradeable Image (Grade 0 Normal)
    img1, fov1, v1, od1, f1 = create_fundus_base(width=512, height=512)
    cv2.imwrite(os.path.join(output_dir, "sample_01_clear.png"), img1)
    
    # Sample 2: Low Contrast / Uneven Illumination (Needs Enhancement)
    img2, _, _, _, _ = create_fundus_base(width=512, height=512)
    # Apply illumination gradient (darker on right)
    h, w, _ = img2.shape
    grad = np.tile(np.linspace(1.2, 0.4, w), (h, 1))[:, :, np.newaxis]
    img2 = np.clip(img2.astype(float) * grad, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_02_low_contrast.png"), img2)
    
    # Sample 3: Blurry / Out of Focus (Ungradeable - Reject)
    img3, _, _, _, _ = create_fundus_base(width=512, height=512, blur_sigma=8.0)
    cv2.imwrite(os.path.join(output_dir, "sample_03_blurry.png"), img3)
    
    # Sample 4: Dark / Under-exposed (Ungradeable - Reject)
    img4, _, _, _, _ = create_fundus_base(width=512, height=512, background_dark=True)
    cv2.imwrite(os.path.join(output_dir, "sample_04_dark.png"), img4)
    
    # Sample 5: Cropped / Incomplete FOV (Ungradeable - Reject)
    img5, _, _, _, _ = create_fundus_base(width=512, height=512, fov_ratio=0.45)
    cv2.imwrite(os.path.join(output_dir, "sample_05_cropped.png"), img5)
    
    # Sample 6: Moderate DR (Grade 2 - Referable) with Microaneurysms & Exudates
    img6, _, _, _, _ = create_fundus_base(width=512, height=512)
    # Add Microaneurysms (small red dots)
    ma_coords = [(280, 200), (310, 220), (330, 180), (250, 300), (360, 260), (290, 340)]
    for pt in ma_coords:
        cv2.circle(img6, pt, 3, (15, 15, 180), -1)
    # Add Exudates (bright yellow waxy spots near macula)
    ex_coords = [(310, 270), (325, 280), (300, 290)]
    for pt in ex_coords:
        cv2.circle(img6, pt, 6, (180, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "sample_06_moderate_dr.png"), img6)

    # Sample 7: Severe DR (Grade 3 - Referable) with Hemorrhages & Exudate Clusters
    img7, _, _, _, _ = create_fundus_base(width=512, height=512)
    # Add Hemorrhages (dark blotches)
    hem_coords = [(260, 180), (380, 220), (220, 350), (400, 320)]
    for pt in hem_coords:
        cv2.ellipse(img7, pt, (10, 6), 30, 0, 360, (5, 10, 90), -1)
    # Add Exudate Clusters
    for pt in [(330, 240), (345, 250), (360, 245), (320, 260)]:
        cv2.circle(img7, pt, 7, (200, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "sample_07_severe_dr.png"), img7)

    # Sample 8: Proliferative DR (Grade 4 - Referable) with Neovascularization
    img8, _, _, od8, _ = create_fundus_base(width=512, height=512)
    # Add Neovascularization fine vessel loops near Optic Disc
    nv_points = np.array([[od8[0]+15, od8[1]-10], [od8[0]+35, od8[1]-25], [od8[0]+45, od8[1]+5], [od8[0]+20, od8[1]+15]], np.int32)
    cv2.polylines(img8, [nv_points], True, (15, 20, 140), 2)
    cv2.imwrite(os.path.join(output_dir, "sample_08_proliferative_dr.png"), img8)

    print(f"Successfully generated 8 sample fundus images in '{output_dir}/'")

if __name__ == "__main__":
    generate_dataset_samples()
