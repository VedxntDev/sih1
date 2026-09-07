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
            'focus_min_pass': 100.0,
            'focus_min_reject': 40.0,
            'fov_min_pass': 0.60,
            'fov_min_reject': 0.40,
            'contrast_min_pass': 26.0,
            'contrast_min_reject': 15.0,
            'illum_std_max_pass': 0.22
        }

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")

    h, w, c = img.shape
    b_chan, g_chan, r_chan = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Field of View (FOV) Mask & Coverage (Normalized against circular aperture for all aspect ratios)
    _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    fov_area = float(np.sum(fov_mask > 0))
    
    # Inscribed circle maximum area for this image frame
    max_circle_area = (np.pi / 4.0) * (min(h, w) ** 2)
    fov_ratio = float(min(1.0, fov_area / (max_circle_area + 1e-5)))

    # Enclosing circular completeness
    contours, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c_max)
        enclosing_area = np.pi * (radius ** 2)
        fov_completeness = float(fov_area / (enclosing_area + 1e-5))
    else:
        fov_completeness = 0.0

    import hashlib

    # 0. Digital Provenance & Cryptographic Fingerprinting (SHA-256)
    with open(img_path, 'rb') as f:
        file_bytes = f.read()
    image_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # 1. Real Fundus Photo Authenticity & Synthetic/Text Overlay Classifier
    # Check 1: Aspect Ratio Gate (Clinical fundus cameras capture in 1:1, 4:3, or 3:2. Multi-panel posters are > 1.65:1)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    is_multipanel_graphic = bool(aspect_ratio > 1.65)

    # Check 2: Multi-Aperture / Multiple Retinal Circle Detection
    contours_all, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_discs = [cnt for cnt in contours_all if cv2.contourArea(cnt) > (0.08 * h * w)]
    has_multiple_discs = bool(len(large_discs) > 1)

    # Check 3: Burned-in Text Labels / Arrow Annotations
    white_text_mask = (r_chan > 230) & (g_chan > 230) & (b_chan > 230)
    num_t, t_labels, t_stats, _ = cv2.connectedComponentsWithStats(white_text_mask.astype(np.uint8))
    char_strokes = sum(1 for i in range(1, num_t) if 10 <= t_stats[i, cv2.CC_STAT_AREA] <= 400 and 6 <= t_stats[i, cv2.CC_STAT_HEIGHT] <= 38)
    has_heavy_text_overlay = bool(char_strokes > 12)

    # Check 4: Retinal Color Gamut (Predominantly reddish-orange: R >= G >= B inside aperture)
    in_aperture = fov_mask > 0
    if np.sum(in_aperture) > 500:
        r_fov = r_chan[in_aperture].astype(float)
        g_fov = g_chan[in_aperture].astype(float)
        b_fov = b_chan[in_aperture].astype(float)
        
        red_dominance = np.mean(r_fov > (g_fov * 0.92))
        blue_suppression = np.mean(g_fov > (b_fov * 0.88))
        is_retinal_gamut = bool(red_dominance > 0.65 and blue_suppression > 0.60)
    else:
        is_retinal_gamut = False

    # Check 5: Laterality Detection (OD: Right Eye, OS: Left Eye based on nasal Optic Disc position)
    rg_composite = cv2.GaussianBlur(r_chan.astype(float)*0.6 + g_chan.astype(float)*0.4, (31, 31), 0)
    rg_composite[fov_mask == 0] = 0
    _, _, _, max_loc = cv2.minMaxLoc(rg_composite)
    od_x, od_y = max_loc
    laterality = "OS (Left Eye)" if od_x < (w // 2) else "OD (Right Eye)"

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

    # 5. Enhancement Sub-pipeline & Statistical Divergence Verification
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)
    enhanced_lab = cv2.merge([l_clahe, a_chan, b_chan])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_img[fov_mask == 0] = 0

    # Calculate CLAHE Chi-Square Histogram Divergence (verifies distinct processing stage)
    hist_raw = cv2.calcHist([l_chan], [0], fov_mask, [64], [0, 256])
    hist_enh = cv2.calcHist([l_clahe], [0], fov_mask, [64], [0, 256])
    cv2.normalize(hist_raw, hist_raw)
    cv2.normalize(hist_enh, hist_enh)
    clahe_chi_sq = float(cv2.compareHist(hist_raw, hist_enh, cv2.HISTCMP_CHISQR))
    is_distinct_enhancement = bool(clahe_chi_sq > 0.04)

    quality_report = {
        'image_sha256': image_sha256,
        'laterality': laterality,
        'is_retinal_gamut': is_retinal_gamut,
        'clahe_chi_sq': clahe_chi_sq,
        'is_distinct_enhancement': is_distinct_enhancement,
        'focus_score': focus_score,
        'fft_focus_score': fft_focus_score,
        'fov_ratio': fov_ratio,
        'fov_completeness': fov_completeness,
        'contrast_score': contrast_score,
        'mean_brightness': mean_brightness,
        'illumination_std': illumination_std
    }

    # 6. Decision Gatekeeping (Multi-Stage Integrity & Quality Filter)
    rejection_reason = ""
    if is_multipanel_graphic:
        status = 'reject'
        rejection_reason = f"Non-Clinical Teaching Infographic Detected (Aspect Ratio {aspect_ratio:.2f}:1 > 1.65:1) — Please upload single unannotated raw camera capture."
    elif has_multiple_discs:
        status = 'reject'
        rejection_reason = f"Multi-Panel Composite Image ({len(large_discs)} eye discs detected) — Please crop or upload a single eye fundus scan."
    elif has_heavy_text_overlay:
        status = 'reject'
        rejection_reason = f"Burned-in Text Labels / Arrow Annotations Detected ({char_strokes} text strokes) — Rejecting educational graphic; raw clinical photo required."
    elif not is_retinal_gamut:
        status = 'reject'
        rejection_reason = "Non-Retinal Image / Invalid Color Gamut Detected — Only authentic fundus photographs are accepted."
    elif fov_ratio < params['fov_min_reject'] and fov_completeness < 0.50:
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

    if status == 'reject':
        enhanced_img = img.copy()

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
