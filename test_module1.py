#!/usr/bin/env python3
"""
Module 1 Test Harness: Image Quality Assessment & Enhancement Engine
Executes quality gatekeeping and enhancement on sample fundus images,
calculating focus, FOV, illumination, and contrast metrics.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assess_and_enhance(img_path, params=None):
    """
    Python companion function matching assessAndEnhance.m
    Computes quality metrics, gatekeeps image gradeability, and applies CLAHE enhancement.
    """
    if params is None:
        params = {
            'focus_min_pass': 110.0,
            'focus_min_reject': 45.0,
            'fov_min_pass': 0.70,
            'fov_min_reject': 0.55,
            'contrast_min_pass': 32.0,
            'contrast_min_reject': 18.0,
            'illum_std_max_pass': 0.16
        }

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")

    h, w, c = img.shape
    b_chan, g_chan, r_chan = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Field of View (FOV) Mask & Coverage
    _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    fov_area = np.sum(fov_mask > 0)
    total_area = h * w
    fov_ratio = fov_area / total_area

    # 2. Focus / Sharpness Metric (Laplacian Variance)
    lap = cv2.Laplacian(g_chan, cv2.CV_64F)
    focus_score = float(np.var(lap[fov_mask > 0])) if fov_area > 0 else float(np.var(lap))

    # 2D FFT spectral high-frequency ratio
    f_transform = np.fft.fftshift(np.fft.fft2(g_chan.astype(float)))
    magnitude = np.abs(f_transform)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    high_freq_mask = dist > (min(h, w) * 0.25)
    fft_focus_score = float(np.sum(magnitude[high_freq_mask]) / (np.sum(magnitude) + 1e-6))

    # 3. Illumination Uniformity (Quadrant Std Dev)
    hh, hw = h // 2, w // 2
    q1 = g_chan[:hh, :hw]
    q2 = g_chan[:hh, hw:]
    q3 = g_chan[hh:, :hw]
    q4 = g_chan[hh:, hw:]
    q_means = [np.mean(q[q > 15]) if np.any(q > 15) else np.mean(g_chan) for q in [q1, q2, q3, q4]]
    illumination_std = float(np.std(q_means) / (np.mean(q_means) + 1e-6))

    # 4. Contrast Score (RMS contrast)
    if fov_area > 0:
        contrast_score = float(np.std(g_chan[fov_mask > 0]))
        mean_brightness = float(np.mean(g_chan[fov_mask > 0]))
    else:
        contrast_score = float(np.std(g_chan))
        mean_brightness = float(np.mean(g_chan))

    quality_report = {
        'focus_score': focus_score,
        'fft_focus_score': fft_focus_score,
        'fov_ratio': fov_ratio,
        'contrast_score': contrast_score,
        'mean_brightness': mean_brightness,
        'illumination_std': illumination_std
    }

    # 5. Decision Gatekeeping
    rejection_reason = ""
    if fov_ratio < params['fov_min_reject']:
        status = 'reject'
        rejection_reason = f"Incomplete Field of View (Coverage: {fov_ratio*100:.1f}%, Min: {params['fov_min_reject']*100:.1f}%) — Re-align fundus camera centered on pupil."
    elif focus_score < params['focus_min_reject']:
        status = 'reject'
        rejection_reason = f"Out of Focus / Severe Blur (Focus Score: {focus_score:.1f}, Min: {params['focus_min_reject']:.1f}) — Adjust camera focus dial before recapture."
    elif contrast_score < params['contrast_min_reject'] or mean_brightness < 20.0:
        status = 'reject'
        rejection_reason = f"Severe Illumination Deficiency (Contrast: {contrast_score:.1f}, Brightness: {mean_brightness:.1f}) — Increase flash illumination."
    elif focus_score < params['focus_min_pass'] or contrast_score < params['contrast_min_pass'] or illumination_std > params['illum_std_max_pass']:
        status = 'enhance'
    else:
        status = 'pass'

    # 6. Enhancement Sub-pipeline
    if status == 'reject':
        enhanced_img = img.copy()
    else:
        # CLAHE on green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g_clahe = clahe.apply(g_chan)

        # Background Illumination Normalization
        bg_blur = cv2.GaussianBlur(g_clahe.astype(float), (61, 61), 15)
        target_mean = np.mean(g_chan[fov_mask > 0]) if fov_area > 0 else np.mean(g_chan)
        g_norm = np.clip((g_clahe.astype(float) / (bg_blur + 1e-5)) * target_mean, 0, 255).astype(np.uint8)

        # Denoising
        g_denoised = cv2.medianBlur(g_norm, 3)

        # Color synthesis in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(float) * 1.15, 0, 255).astype(np.uint8) # Saturation boost
        hsv[:, :, 2] = g_denoised # Replace Value with enhanced green channel
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply FOV mask
        enhanced_img[fov_mask == 0] = 0

    return status, enhanced_img, quality_report, rejection_reason

def run_module1_harness(input_dir="data/sample_images", output_dir="output/module1"):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print("=" * 90)
    print(" MODULE 1: IMAGE QUALITY ASSESSMENT & ENHANCEMENT HARNESS")
    print("=" * 90)
    print(f"{'Image File':<28} | {'Status':<8} | {'Focus':<7} | {'FOV %':<6} | {'Contrast':<8} | {'Illum Std':<9} | {'Action / Rationale'}")
    print("-" * 90)

    summary_records = []

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        status, enhanced_img, q_report, reason = assess_and_enhance(img_path)
        
        orig_img = cv2.imread(img_path)

        # Save side-by-side output for visualization
        comp = np.hstack([orig_img, enhanced_img])
        cv2.putText(comp, f"Original: {img_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(comp, f"Status: {status.upper()}", (w_comp := comp.shape[1]//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0,255,0) if status=='pass' else (0,255,255) if status=='enhance' else (0,0,255), 2)

        save_path = os.path.join(output_dir, f"module1_result_{img_name}")
        cv2.imwrite(save_path, comp)

        action_msg = reason if status == 'reject' else ("Enhanced via CLAHE + Illum Norm" if status == 'enhance' else "Passed — Gradeable")
        print(f"{img_name:<28} | {status.upper():<8} | {q_report['focus_score']:<7.1f} | {q_report['fov_ratio']*100:<6.1f} | {q_report['contrast_score']:<8.1f} | {q_report['illumination_std']:<9.3f} | {action_msg}")

        summary_records.append({
            'name': img_name,
            'status': status,
            'report': q_report,
            'reason': reason,
            'orig': orig_img,
            'enhanced': enhanced_img
        })

    print("-" * 90)
    print(f"Module 1 evaluation complete. Output visual comparisons saved to '{output_dir}/'")

    # Plot summary dashboard figure
    fig, axes = plt.subplots(len(summary_records), 2, figsize=(10, 2.5 * len(summary_records)))
    fig.suptitle("Module 1: Image Quality Assessment & CLAHE Enhancement Results", fontsize=14, fontweight='bold')

    for idx, rec in enumerate(summary_records):
        orig_rgb = cv2.cvtColor(rec['orig'], cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(rec['enhanced'], cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(orig_rgb)
        axes[idx, 0].set_title(f"Input: {rec['name']}\nFocus: {rec['report']['focus_score']:.1f}, FOV: {rec['report']['fov_ratio']*100:.0f}%", fontsize=9)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(enh_rgb)
        color = 'green' if rec['status'] == 'pass' else 'orange' if rec['status'] == 'enhance' else 'red'
        axes[idx, 1].set_title(f"Output: [{rec['status'].upper()}]\n{rec['reason'] if rec['reason'] else 'Gradeable & Enhanced'}", fontsize=9, color=color)
        axes[idx, 1].axis('off')

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "module1_quality_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary quality dashboard saved to '{dashboard_path}'")

if __name__ == "__main__":
    run_module1_harness()
