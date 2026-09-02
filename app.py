#!/usr/bin/env python3
"""
SIH 2026 Explainable AI Diabetic Retinopathy Screening — Web Application Server
Provides a complete web interface for users to upload fundus images, execute Modules 1-5,
and inspect visual overlays, Grad-CAM heatmaps, severity scores, and clinical reports.
"""

import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

from test_module1 import assess_and_enhance
from test_module2 import segment_retinal_structures
from test_module3 import grade_dr
from test_module4 import explain_prediction
from test_module5 import simulate_telemedicine_queue

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def sanitize_for_json(obj):
    """Recursively converts NumPy datatypes (int64, float64, bool_) to native Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

def image_to_base64(img_bgr):
    """Converts OpenCV BGR image to base64 JPEG string for inline HTML rendering."""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainable AI for Diabetic Retinopathy Screening (SIH 2026)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1a5276;
            --primary-light: #2980b9;
            --accent: #27ae60;
            --danger: #e74c3c;
            --warning: #f39c12;
            --bg: #f4f7f9;
            --card-bg: #ffffff;
            --text: #2c3e50;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { margin: 0; padding: 0; background-color: var(--bg); color: var(--text); }
        header {
            background: linear-gradient(135deg, #1b365d, #2980b9);
            color: white; padding: 20px 40px;
            display: flex; justify-content: space-between; align-items: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        header h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
        header p { margin: 4px 0 0 0; opacity: 0.85; font-size: 13px; }
        .badge-sih { background: #f39c12; color: #fff; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 12px; }
        
        .container { display: grid; grid-template-columns: 340px 1fr; gap: 25px; max-width: 1400px; margin: 30px auto; padding: 0 25px; }
        
        .panel { background: var(--card-bg); border-radius: 12px; padding: 22px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid var(--border); }
        .panel h2 { margin-top: 0; font-size: 17px; color: var(--primary); border-bottom: 2px solid #edf2f7; padding-bottom: 10px; }
        
        .upload-area {
            border: 2px dashed #cbd5e0; border-radius: 10px; padding: 30px 15px; text-align: center;
            background: #fafafa; cursor: pointer; transition: all 0.2s;
        }
        .upload-area:hover { border-color: var(--primary-light); background: #f0f7ff; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 38px; color: #a0aec0; margin-bottom: 8px; }
        
        .btn-screen {
            width: 100%; padding: 13px; background: var(--primary-light); color: white; border: none;
            border-radius: 8px; font-weight: 600; font-size: 15px; cursor: pointer; margin-top: 15px;
            transition: 0.2s; box-shadow: 0 4px 10px rgba(41,128,185,0.3);
        }
        .btn-screen:hover { background: #1f618d; }
        .btn-screen:disabled { background: #a0aec0; cursor: not-allowed; }
        
        .sample-presets { margin-top: 20px; }
        .sample-presets p { font-size: 12px; font-weight: 600; color: #718096; margin-bottom: 8px; text-transform: uppercase; }
        .sample-btn {
            display: block; width: 100%; text-align: left; padding: 8px 12px; margin-bottom: 6px;
            background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 13px; cursor: pointer;
            transition: 0.2s;
        }
        .sample-btn:hover { background: #edf2f7; border-color: #cbd5e0; }

        .result-card { background: white; border-radius: 10px; padding: 18px; margin-bottom: 20px; border-left: 5px solid var(--primary-light); box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .status-header { display: flex; justify-content: space-between; align-items: center; }
        .status-badge { padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 13px; color: white; }
        
        .grid-2x2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
        
        .img-card { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }
        .img-card img { width: 100%; height: 210px; object-fit: contain; border-radius: 6px; }
        .img-card p { margin: 8px 0 0 0; font-size: 12px; font-weight: 600; color: #4a5568; }

        .metric-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; background: #f8fafc; padding: 12px; border-radius: 8px; margin-top: 12px; }
        .metric-item { text-align: center; }
        .metric-value { font-size: 18px; font-weight: 700; color: var(--primary); }
        .metric-label { font-size: 11px; color: #718096; text-transform: uppercase; }

        .rationale-box { background: #2d3748; color: #edf2f7; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; line-height: 1.5; margin-top: 12px; white-space: pre-wrap; }

        .action-bar { display: flex; gap: 10px; margin-top: 15px; }
        .btn-action { padding: 9px 16px; border: none; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer; color: white; }
        .btn-approve { background: var(--accent); }
        .btn-reject { background: var(--warning); }
        .btn-escalate { background: #8e44ad; }

        #spinner { display: none; text-align: center; padding: 40px; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid var(--primary-light); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Explainable AI for Diabetic Retinopathy Screening</h1>
            <p>Integrated MATLAB/Python Tele-Ophthalmology Prototype (Modules 1 to 5)</p>
        </div>
        <span class="badge-sih">SIH 2026 | PS ID 26038</span>
    </header>

    <div class="container">
        <!-- Control Panel -->
        <div class="panel">
            <h2>1. Upload Fundus Image</h2>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📷</div>
                <strong style="font-size:14px;">Click or Drag Image Here</strong>
                <p style="font-size:11px; color:#718096; margin-top:4px;">Supports PNG, JPG, JPEG fundus images</p>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            <div id="previewName" style="font-size:12px; color:var(--primary); font-weight:600; margin-top:8px; text-align:center;"></div>

            <button class="btn-screen" id="btnScreen" onclick="runScreening()" disabled>🚀 Run AI Screening Pipeline</button>

            <div class="sample-presets">
                <p>Or Select Sample Test Case:</p>
                <button class="sample-btn" onclick="selectSample('sample_01_clear.png')">🟢 Grade 0 Normal (Clear)</button>
                <button class="sample-btn" onclick="selectSample('sample_02_low_contrast.png')">🟡 Low Contrast (Needs CLAHE)</button>
                <button class="sample-btn" onclick="selectSample('sample_03_blurry.png')">🔴 Blurry Image (Gatekeeper Reject)</button>
                <button class="sample-btn" onclick="selectSample('sample_06_moderate_dr.png')">🟠 Grade 2 Moderate DR (Referable)</button>
                <button class="sample-btn" onclick="selectSample('sample_07_severe_dr.png')">🔴 Grade 3 Severe DR (Hemorrhages)</button>
                <button class="sample-btn" onclick="selectSample('sample_08_proliferative_dr.png')">🟣 Grade 4 Proliferative (Neovascular)</button>
            </div>
        </div>

        <!-- Output Display Panel -->
        <div class="panel">
            <h2>2. Diagnostic Screening Results</h2>

            <div id="placeholder">
                <p style="text-align:center; color:#a0aec0; margin-top:80px; font-size:15px;">
                    👈 Upload an eye image or pick a sample case on the left to view full 5-module AI diagnostics.
                </p>
            </div>

            <div id="spinner">
                <div class="loader"></div>
                <p style="font-weight:600; color:var(--primary);">Processing Modules 1-5 (Quality, Segmentation, Grading, Grad-CAM)...</p>
            </div>

            <div id="resultsContent" style="display:none;">
                <!-- Summary Card -->
                <div class="result-card" id="summaryCard">
                    <div class="status-header">
                        <div>
                            <h3 id="resGradeName" style="margin:0; font-size:20px; color:var(--primary);">Grade 2 Moderate DR</h3>
                            <p id="resConfidence" style="margin:4px 0 0 0; font-size:13px; color:#718096;">Confidence Score: 88.5%</p>
                        </div>
                        <span class="status-badge" id="resBadge">REFERRAL REQUIRED</span>
                    </div>
                </div>

                <!-- Image Quad View -->
                <div class="grid-4">
                    <div class="img-card">
                        <img id="imgOrig" src="" alt="Original">
                        <p>1. Original Fundus</p>
                    </div>
                    <div class="img-card">
                        <img id="imgEnhanced" src="" alt="Enhanced">
                        <p>2. CLAHE Enhanced (Mod 1)</p>
                    </div>
                    <div class="img-card">
                        <img id="imgOverlay" src="" alt="Overlay">
                        <p>3. Lesion Overlay (Mod 2)</p>
                    </div>
                    <div class="img-card">
                        <img id="imgGradCAM" src="" alt="GradCAM">
                        <p>4. Grad-CAM Explainability (Mod 4)</p>
                    </div>
                </div>

                <!-- Biomarker Telemetry -->
                <div class="metric-list">
                    <div class="metric-item">
                        <div class="metric-value" id="valMAs">0</div>
                        <div class="metric-label">Microaneurysms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="valExudates">0</div>
                        <div class="metric-label">Exudates</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="valHems">0</div>
                        <div class="metric-label">Hemorrhages</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="valFocus">0</div>
                        <div class="metric-label">Focus Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="valOverlap">0.0</div>
                        <div class="metric-label">Grad-CAM Overlap</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="valNV">No</div>
                        <div class="metric-label">Neovascularization</div>
                    </div>
                </div>

                <!-- Clinical Rationale -->
                <div class="rationale-box" id="resRationale"></div>

                <!-- Doctor Actions -->
                <div class="action-bar">
                    <button class="btn-action btn-approve" onclick="alert('Case Approved by Doctor!')">✓ Approve AI Diagnosis (<30s)</button>
                    <button class="btn-action btn-reject" onclick="alert('Case Flagged for Re-grading!')">✗ Override Grade</button>
                    <button class="btn-action btn-escalate" onclick="alert('Escalated to Retina Specialist!')">⚑ Escalate Specialist</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        let selectedSampleName = null;

        function handleFileSelect(event) {
            const files = event.target.files;
            if (files.length > 0) {
                selectedFile = files[0];
                selectedSampleName = null;
                document.getElementById('previewName').innerText = "Selected: " + selectedFile.name;
                document.getElementById('btnScreen').disabled = false;
            }
        }

        function selectSample(sampleName) {
            selectedSampleName = sampleName;
            selectedFile = null;
            document.getElementById('previewName').innerText = "Selected Preset: " + sampleName;
            document.getElementById('btnScreen').disabled = false;
            runScreening();
        }

        function runScreening() {
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('resultsContent').style.display = 'none';
            document.getElementById('spinner').style.display = 'block';

            const formData = new FormData();
            if (selectedFile) {
                formData.append('file', selectedFile);
            } else if (selectedSampleName) {
                formData.append('sample_name', selectedSampleName);
            }

            fetch('/api/screen', {
                method: 'POST',
                body: formData
            })
            .then(async res => {
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error("Server error (" + res.status + "): " + text);
                }
                return res.json();
            })
            .then(data => {
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('resultsContent').style.display = 'block';

                document.getElementById('resGradeName').innerText = data.grade_name;
                document.getElementById('resConfidence').innerText = "Confidence Score: " + (data.confidence * 100).toFixed(1) + "%";

                const badge = document.getElementById('resBadge');
                if (data.status === 'reject') {
                    badge.innerText = "GATEKEEPER REJECTED";
                    badge.style.backgroundColor = "#e74c3c";
                } else if (data.referable) {
                    badge.innerText = "REFERRAL REQUIRED";
                    badge.style.backgroundColor = "#e74c3c";
                } else {
                    badge.innerText = "ROUTINE / CLEAR";
                    badge.style.backgroundColor = "#27ae60";
                }

                document.getElementById('imgOrig').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('imgEnhanced').src = "data:image/jpeg;base64," + data.img_enhanced;
                document.getElementById('imgOverlay').src = "data:image/jpeg;base64," + data.img_overlay;
                document.getElementById('imgGradCAM').src = "data:image/jpeg;base64," + data.img_gradcam;

                document.getElementById('valMAs').innerText = data.stats.ma_count;
                document.getElementById('valExudates').innerText = data.stats.exudate_count;
                document.getElementById('valHems').innerText = data.stats.hem_count;
                document.getElementById('valFocus').innerText = data.quality.focus_score.toFixed(1);
                document.getElementById('valOverlap').innerText = data.correlation_score.toFixed(2);
                document.getElementById('valNV').innerText = data.stats.nv_flag ? "YES" : "No";

                document.getElementById('resRationale').innerText = data.rationale;
            })
            .catch(err => {
                document.getElementById('spinner').style.display = 'none';
                alert("Error running screening pipeline: " + err.message);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/screen', methods=['POST'])
def api_screen():
    try:
        file = request.files.get('file')
        sample_name = request.form.get('sample_name')

        img_path = None
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
        elif sample_name:
            img_path = os.path.join('data/sample_images', sample_name)

        if not img_path or not os.path.exists(img_path):
            return jsonify({'error': 'No image provided or file not found'}), 400

        img_orig = cv2.imread(img_path)
        if img_orig is None:
            return jsonify({'error': f'Invalid image format: could not decode {img_path}'}), 400

        # 1. Module 1: Quality Gatekeeper & Enhancement
        status, enhanced, q_report, reason = assess_and_enhance(img_path)

        # 2. Module 2: Structure & Lesion Segmentation
        overlay, stats, masks = segment_retinal_structures(enhanced)

        # 3. Module 3: DR Severity Grading
        level, ref, conf, probs, ref_prob = grade_dr(stats)

        # 4. Module 4: Explainability & Grad-CAM
        heatmap, corr_score, report = explain_prediction(enhanced, level, ref, conf, stats, masks)

        if status == 'reject':
            rationale = f"[QUALITY GATEKEEPER REJECTED]\nReason: {reason}\nAction: Please adjust camera focus or illumination and recapture."
        else:
            rationale = report['rationale_text']

        # Sanitize all data structures for clean JSON serialization
        response_data = sanitize_for_json({
            'status': status,
            'grade_level': level,
            'grade_name': report['severity_name'],
            'referable': ref,
            'confidence': conf,
            'quality': q_report,
            'stats': stats,
            'correlation_score': corr_score,
            'rationale': rationale,
            'img_orig': image_to_base64(img_orig),
            'img_enhanced': image_to_base64(enhanced),
            'img_overlay': image_to_base64(overlay),
            'img_gradcam': image_to_base64(heatmap)
        })

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting SIH 2026 DR Screening Web Server on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
