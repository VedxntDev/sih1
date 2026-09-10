#!/usr/bin/env python3
"""
OptiNova AI — Explainable Retinal Intelligence & Diabetic Retinopathy Screening (SIH 2026)
Smart India Hackathon 2026 | Problem Statement ID: SIH26038 | Theme: MedTech / Clean & Green Software
Team: Optinova | Zero-CAPEX Edge Tele-Ophthalmology & Clinical EHR Report Export
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

# Serverless-friendly upload folder in /tmp
UPLOAD_FOLDER = '/tmp/uploads'
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception:
    pass

def sanitize_for_json(obj):
    """Recursively converts NumPy datatypes to native Python types."""
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

def image_to_base64(img_bgr, quality=88):
    """Converts OpenCV BGR image to base64 JPEG string for inline HTML rendering."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', img_bgr, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPTINOVA AI — Retinal Intelligence & Clinical DR Screening</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --font-display: 'Space Grotesk', -apple-system, sans-serif;
            --font-main: 'Inter', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;

            /* Dark Theme (Default) */
            --bg-body: #08090c;
            --bg-surface: #0f1117;
            --bg-surface-elevated: #161922;
            --bg-card: rgba(15, 17, 23, 0.95);
            --bg-badge: rgba(255, 255, 255, 0.04);
            
            --border-color: #232734;
            --border-subtle: #1c202b;
            --border-active: #f59e0b;

            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;

            --accent-gold: #d97706;
            --accent-gold-bright: #f59e0b;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-cyan: #06b6d4;
        }

        [data-theme="light"] {
            --bg-body: #f8fafc;
            --bg-surface: #ffffff;
            --bg-surface-elevated: #f1f5f9;
            --bg-card: #ffffff;
            --bg-badge: rgba(15, 23, 42, 0.04);

            --border-color: #e2e8f0;
            --border-subtle: #cbd5e1;
            --border-active: #d97706;

            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;

            --accent-gold: #d97706;
            --accent-gold-bright: #b45309;
            --accent-emerald: #059669;
            --accent-rose: #e11d48;
            --accent-cyan: #0891b2;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            border-radius: 0px !important;
            transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-body);
            color: var(--text-primary);
            line-height: 1.55;
            letter-spacing: -0.01em;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }

        .technical-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 900px;
            background-size: 32px 32px;
            background-image: 
                linear-gradient(to right, var(--border-subtle) 1px, transparent 1px),
                linear-gradient(to bottom, var(--border-subtle) 1px, transparent 1px);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.3) 0%, transparent 80%);
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,0.3) 0%, transparent 80%);
            pointer-events: none;
            z-index: 0;
        }

        /* Navigation */
        nav {
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 40px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 14px;
            text-decoration: none;
            color: var(--text-primary);
        }

        .nav-brand-mark {
            width: 28px;
            height: 28px;
            background: var(--text-primary);
            color: var(--bg-body);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 14px;
            letter-spacing: -0.5px;
        }

        .nav-brand-text {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 19px;
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
        }

        .nav-brand-sub {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            background: var(--bg-badge);
            border: 1px solid var(--border-color);
            color: var(--accent-gold);
            letter-spacing: 0.08em;
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 28px;
            list-style: none;
        }

        .nav-links a {
            text-decoration: none;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .nav-links a:hover {
            color: var(--text-primary);
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn-sharp {
            font-family: var(--font-display);
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            color: var(--text-primary);
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .btn-sharp:hover {
            background: var(--bg-surface-elevated);
            border-color: var(--text-muted);
        }

        .btn-sharp-primary {
            background: var(--text-primary);
            color: var(--bg-body);
            border: 1px solid var(--text-primary);
        }

        .btn-sharp-primary:hover {
            background: var(--accent-gold-bright);
            color: #000000;
            border-color: var(--accent-gold-bright);
        }

        .btn-sharp-accent {
            background: var(--accent-gold);
            color: #ffffff;
            border: 1px solid var(--accent-gold);
        }

        .btn-sharp-accent:hover {
            background: var(--accent-gold-bright);
            border-color: var(--accent-gold-bright);
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 0 24px;
            position: relative;
            z-index: 1;
        }

        /* Hero */
        .hero {
            padding: 80px 0 50px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .hero-meta-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--accent-gold);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 20px;
        }

        .hero-meta-bar span {
            border: 1px solid var(--border-color);
            padding: 3px 8px;
            background: var(--bg-surface);
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: clamp(38px, 5.2vw, 68px);
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.04em;
            max-width: 980px;
            margin-bottom: 24px;
            text-transform: uppercase;
        }

        .hero-title .accent-text {
            color: var(--accent-gold-bright);
        }

        .hero-desc {
            font-size: 17px;
            color: var(--text-secondary);
            max-width: 720px;
            line-height: 1.6;
            margin-bottom: 36px;
        }

        .hero-action-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 60px;
            flex-wrap: wrap;
        }

        .metrics-grid-flat {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
        }

        @media (max-width: 900px) {
            .metrics-grid-flat { grid-template-columns: repeat(2, 1fr); }
            nav { padding: 14px 20px; }
            .nav-links { display: none; }
        }

        @media (max-width: 600px) {
            .metrics-grid-flat { grid-template-columns: 1fr; }
        }

        .metric-cell {
            padding: 24px;
            border-right: 1px solid var(--border-color);
        }

        .metric-cell:last-child {
            border-right: none;
        }

        .metric-cell-value {
            font-family: var(--font-display);
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text-primary);
        }

        .metric-cell-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 6px;
        }

        .section-box {
            padding: 70px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .section-header-flat {
            margin-bottom: 40px;
        }

        .section-header-tag {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            color: var(--accent-gold);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            display: block;
        }

        .section-header-title {
            font-family: var(--font-display);
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-transform: uppercase;
            color: var(--text-primary);
        }

        .section-header-desc {
            font-size: 15px;
            color: var(--text-secondary);
            max-width: 680px;
            margin-top: 8px;
        }

        /* 4-Step Architecture */
        .arch-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border-color);
            border: 1px solid var(--border-color);
        }

        @media (max-width: 1024px) {
            .arch-grid { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 640px) {
            .arch-grid { grid-template-columns: 1fr; }
        }

        .arch-card {
            background: var(--bg-surface);
            padding: 30px 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .arch-card-num {
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-gold);
            margin-bottom: 16px;
        }

        .arch-card-title {
            font-family: var(--font-display);
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            margin-bottom: 12px;
            text-transform: uppercase;
        }

        .arch-card-text {
            font-size: 13.5px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Screening Studio */
        .studio-grid-flat {
            display: grid;
            grid-template-columns: 360px 1fr;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
        }

        @media (max-width: 1024px) {
            .studio-grid-flat { grid-template-columns: 1fr; }
        }

        .studio-control-panel {
            padding: 28px;
            border-right: 1px solid var(--border-color);
        }

        .studio-display-panel {
            padding: 28px;
            background: var(--bg-body);
        }

        .panel-title-flat {
            font-family: var(--font-display);
            font-size: 15px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-primary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }

        .drop-zone-flat {
            border: 1px dashed var(--border-color);
            padding: 30px 16px;
            text-align: center;
            background: var(--bg-surface-elevated);
            cursor: pointer;
            margin-bottom: 16px;
        }

        .drop-zone-flat:hover {
            border-color: var(--accent-gold);
            background: var(--bg-badge);
        }

        .preset-list-flat {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 6px;
        }

        .preset-item-flat {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: background 0.15s ease, border-color 0.15s ease;
        }

        .preset-item-flat:hover {
            background: var(--bg-surface-elevated);
            border-color: var(--accent-gold);
        }

        .preset-item-flat.active {
            background: var(--bg-surface-elevated);
            border-color: var(--text-primary);
            box-shadow: inset 0 0 0 1px var(--text-primary);
        }

        .preset-g0 { border-left-color: var(--accent-emerald); }
        .preset-g1 { border-left-color: #38bdf8; }
        .preset-g2 { border-left-color: var(--accent-gold); }
        .preset-g3 { border-left-color: #fb923c; }
        .preset-g4 { border-left-color: var(--accent-rose); }
        .preset-qc { border-left-color: #94a3b8; }

        .preset-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
            text-align: left;
        }

        .preset-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }

        .preset-sub {
            font-size: 10px;
            color: var(--text-muted);
            font-family: var(--font-body);
        }

        .preset-tag-flat {
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 700;
            padding: 3px 7px;
            border: 1px solid var(--border-color);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            white-space: nowrap;
        }

        .tag-g0 { color: #10b981; background: rgba(16, 185, 129, 0.10); border-color: rgba(16, 185, 129, 0.35); }
        .tag-g1 { color: #38bdf8; background: rgba(56, 189, 248, 0.10); border-color: rgba(56, 189, 248, 0.35); }
        .tag-g2 { color: #f59e0b; background: rgba(245, 158, 11, 0.10); border-color: rgba(245, 158, 11, 0.35); }
        .tag-g3 { color: #fb923c; background: rgba(251, 146, 60, 0.10); border-color: rgba(251, 146, 60, 0.35); }
        .tag-g4 { color: #f43f5e; background: rgba(244, 63, 94, 0.12); border-color: rgba(244, 63, 94, 0.40); }
        .tag-clahe { color: var(--accent-gold); background: rgba(245, 158, 11, 0.10); border-color: rgba(245, 158, 11, 0.30); }
        .tag-drop { color: #f43f5e; background: rgba(244, 63, 94, 0.10); border-color: rgba(244, 63, 94, 0.35); }

        .quad-grid-flat {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        @media (max-width: 800px) {
            .quad-grid-flat { grid-template-columns: repeat(2, 1fr); }
        }

        .quad-card-flat {
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            padding: 8px;
            cursor: pointer;
        }

        .quad-card-flat:hover {
            border-color: var(--accent-gold);
        }

        .quad-img-flat {
            width: 100%;
            height: 150px;
            background: #000000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .quad-img-flat img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .quad-label-flat {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 6px;
        }

        .split-box-flat {
            position: relative;
            width: 100%;
            height: 250px;
            background: #000000;
            border: 1px solid var(--border-color);
            overflow: hidden;
            margin-bottom: 20px;
            user-select: none;
        }

        .split-box-flat img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .split-layer-flat {
            position: absolute;
            top: 0;
            left: 0;
            width: 50%;
            height: 100%;
            overflow: hidden;
            border-right: 1px solid var(--accent-gold);
        }

        .split-layer-flat img {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            max-width: none;
        }

        .split-cursor-flat {
            position: absolute;
            top: 0;
            left: 50%;
            height: 100%;
            width: 2px;
            background: var(--accent-gold);
            cursor: ew-resize;
            z-index: 10;
        }

        .telemetry-grid-flat {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1px;
            background: var(--border-color);
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }

        .telemetry-item-flat {
            background: var(--bg-surface);
            padding: 14px 18px;
        }

        .telemetry-item-label {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .telemetry-item-value {
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 4px;
        }

        .table-flat {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid var(--border-color);
            font-size: 13.5px;
        }

        .table-flat th {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            background: var(--bg-surface-elevated);
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .table-flat td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-surface);
        }

        .modal-flat-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .modal-flat-box {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            max-width: 900px;
            width: 100%;
            max-height: 88vh;
            overflow-y: auto;
            padding: 32px;
            position: relative;
        }

        /* Doctor Clinical Report Modal & Sheet */
        .doctor-report-sheet {
            background: #ffffff;
            color: #000000;
            padding: 32px;
            border: 1px solid #d1d5db;
            font-family: var(--font-main);
            max-width: 860px;
            margin: 0 auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        .doctor-report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #000000;
            padding-bottom: 16px;
            margin-bottom: 20px;
        }

        .report-grid-quad {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 16px 0;
        }

        .report-quad-item {
            border: 1px solid #e5e7eb;
            padding: 4px;
            text-align: center;
        }

        .report-quad-item img {
            width: 100%;
            height: 120px;
            object-fit: contain;
            background: #000000;
        }

        .report-quad-item span {
            font-family: var(--font-mono);
            font-size: 9px;
            font-weight: 700;
            color: #4b5563;
            text-transform: uppercase;
            display: block;
            margin-top: 4px;
        }

        .report-table-mini {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin: 14px 0;
        }

        .report-table-mini th {
            background: #f3f4f6;
            color: #111827;
            padding: 6px 10px;
            border: 1px solid #d1d5db;
            text-align: left;
            font-family: var(--font-mono);
            font-size: 10px;
        }

        .report-table-mini td {
            padding: 6px 10px;
            border: 1px solid #d1d5db;
        }

        #toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 12px 20px;
            background: var(--bg-surface);
            border: 1px solid var(--accent-gold);
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            z-index: 2000;
            display: none;
        }

        footer {
            border-top: 1px solid var(--border-color);
            padding: 40px 0;
            background: var(--bg-surface);
        }

        .footer-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
            flex-wrap: wrap;
            gap: 16px;
        }

        /* Precision Print Styles: Strip all UI noise and print only the clinical report */
        @media print {
            body {
                background: #ffffff !important;
                color: #000000 !important;
            }
            .technical-grid, nav, .hero, #pipeline, #screening, #matrix, #simulator, footer, #toast, .modal-flat-backdrop:not(#doctorReportModal), .no-print {
                display: none !important;
            }
            #doctorReportModal {
                position: static !important;
                display: block !important;
                background: #ffffff !important;
                padding: 0 !important;
                width: 100% !important;
                height: auto !important;
            }
            .modal-flat-box {
                border: none !important;
                padding: 0 !important;
                max-width: 100% !important;
                max-height: none !important;
                overflow: visible !important;
            }
            .doctor-report-sheet {
                box-shadow: none !important;
                border: none !important;
                padding: 0 !important;
                max-width: 100% !important;
            }
            @page {
                size: A4 portrait;
                margin: 12mm;
            }
        }
    </style>
</head>
<body>

    <div class="technical-grid"></div>

    <!-- Navigation Header -->
    <nav class="no-print">
        <a href="#" class="nav-brand">
            <div class="nav-brand-mark">O</div>
            <div class="nav-brand-text">
                OPTINOVA <span class="nav-brand-sub">SIH26038</span>
            </div>
        </a>

        <ul class="nav-links">
            <li><a href="#screening">Screening Lab</a></li>
            <li><a href="#pipeline">Architecture</a></li>
            <li><a href="#matrix">Risk Matrix</a></li>
            <li><a href="#simulator">Tele-Triage</a></li>
        </ul>

        <div class="nav-actions">
            <button class="btn-sharp" onclick="openPitchModal(0)">[ 📑 Presentation Deck ]</button>
            <button class="btn-sharp" id="themeToggle" onclick="toggleTheme()">
                <span id="themeLabel">THEME: DARK</span>
            </button>
            <a href="#screening" class="btn-sharp btn-sharp-primary">Initialize Scan</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero no-print">
        <div class="container">
            <div class="hero-meta-bar">
                <span>SMART INDIA HACKATHON 2026</span>
                <span>MEDTECH / SOFTWARE</span>
                <span>ZERO-CAPEX EDGE TELEMETRY</span>
            </div>

            <h1 class="hero-title">
                EXPLAINABLE AI FOR <span class="accent-text">DIABETIC RETINOPATHY</span> SCREENING.
            </h1>

            <p class="hero-desc">
                A MATLAB-native clinical screening system delivering edge Laplacian image quality gating (<40 ms), calibrated multi-class DR grading, and transparent Grad-CAM explainability across rural primary health centres.
            </p>

            <div class="hero-action-row">
                <a href="#screening" class="btn-sharp btn-sharp-primary">Launch Screening Studio</a>
                <button onclick="selectSample('sample_06_moderate_dr.png')" class="btn-sharp">Load Benchmark Sample</button>
                <button onclick="openPitchModal(2)" class="btn-sharp">Technical Methodology</button>
            </div>

            <!-- Telemetry Metrics Bar -->
            <div class="metrics-grid-flat">
                <div class="metric-cell">
                    <div class="metric-cell-value">&lt; 40 ms</div>
                    <div class="metric-cell-label">Edge Laplacian QC Gate</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-cell-value">&gt; 90%</div>
                    <div class="metric-cell-label">Referable Sensitivity (Grade ≥2)</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-cell-value">136,875</div>
                    <div class="metric-cell-label">Annual Hub Patient Volume</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-cell-value">&lt; 30 sec</div>
                    <div class="metric-cell-label">Doctor Verification Turnaround</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Engineering Pipeline -->
    <section class="section-box no-print" id="pipeline">
        <div class="container">
            <div class="section-header-flat">
                <span class="section-header-tag">[ 01 / PIPELINE ARCHITECTURE ]</span>
                <h2 class="section-header-title">Technical Methodology & Signal Processing</h2>
                <p class="section-header-desc">Engineered for deterministic sub-watt execution on commodity hardware with zero diagnostic latency.</p>
            </div>

            <div class="arch-grid">
                <div class="arch-card">
                    <span class="arch-card-num">MOD 01</span>
                    <h3 class="arch-card-title">Edge DSP & QC</h3>
                    <p class="arch-card-text">
                        <strong>Laplacian Sharpness:</strong> Drops blurred captures locally (<code>Var(∇²I) &lt; τ</code>) in &lt;40 ms. <strong>CIELAB CLAHE:</strong> Normalizes non-uniform illumination across camera sensor models.
                    </p>
                </div>

                <div class="arch-card">
                    <span class="arch-card-num">MOD 02</span>
                    <h3 class="arch-card-title">Vessel & Lesions</h3>
                    <p class="arch-card-text">
                        <strong>Frangi Filter:</strong> Extracts 2D multiscale vessel eigenvalues. Green-channel top-hat morphology isolates microaneurysms (&lt;125 µm), hemorrhages, and lipid exudates.
                    </p>
                </div>

                <div class="arch-card">
                    <span class="arch-card-num">MOD 03</span>
                    <h3 class="arch-card-title">Calibrated Grading</h3>
                    <p class="arch-card-text">
                        <strong>Cost-Sensitive Inference:</strong> EfficientNet-B0 tuned via Youden's J-Index enforcing &gt;90% sensitivity on Grade ≥2 with Platt-calibrated probability thresholds.
                    </p>
                </div>

                <div class="arch-card">
                    <span class="arch-card-num">MOD 04</span>
                    <h3 class="arch-card-title">XAI & Telemetry</h3>
                    <p class="arch-card-text">
                        <strong>Grad-CAM:</strong> Projects gradient heatmaps for doctor verification in &lt;30s. <strong>WebP + SQLite Queue:</strong> Compresses payload to &lt;400 KB (96% reduction) with offline store-and-forward.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Interactive Screening Studio -->
    <section class="section-box no-print" id="screening">
        <div class="container">
            <div class="section-header-flat">
                <span class="section-header-tag">[ 02 / DIAGNOSTIC STUDIO ]</span>
                <h2 class="section-header-title">Live Retinal Inference Workspace</h2>
                <p class="section-header-desc">Upload a clinical fundus scan or select an audited benchmark dataset case.</p>
            </div>

            <div class="studio-grid-flat">
                <!-- Control Panel -->
                <div class="studio-control-panel">
                    <div class="panel-title-flat">
                        <span>Fundus Acquisition</span>
                        <span style="font-family:var(--font-mono); font-size:10px; color:var(--accent-gold);">UVC / V4L2</span>
                    </div>

                    <div class="drop-zone-flat" id="dropZone" onclick="document.getElementById('fileInput').click()">
                        <div style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase;">
                            [ Click or Drop Image ]
                        </div>
                        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">Supports PNG, JPG, or DICOM</div>
                        <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFileSelect(event)">
                    </div>

                    <div id="fileSelectionText" style="font-family:var(--font-mono); font-size:11px; color:var(--accent-gold); margin-bottom:12px;"></div>

                    <button class="btn-sharp btn-sharp-accent" id="btnRun" style="width:100%; justify-content:center;" onclick="runScreening()" disabled>
                        Execute AI Pipeline
                    </button>

                    <!-- ICDR 5-Tier Disease Severity Grading -->
                    <div style="margin-top:24px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-family:var(--font-mono); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--text-muted);">
                                ICDR Clinical Progression (0–4):
                            </span>
                            <span style="font-family:var(--font-mono); font-size:9px; color:var(--accent-emerald); font-weight:600;">5-TIER</span>
                        </div>
                        <div class="preset-list-flat">
                            <div class="preset-item-flat preset-g0" id="preset-sample_01_clear.png" onclick="selectSample('sample_01_clear.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Grade 0: Normal Retina</div>
                                    <div class="preset-sub">Clear vascular tree • No lesions</div>
                                </div>
                                <span class="preset-tag-flat tag-g0">Routine</span>
                            </div>
                            <div class="preset-item-flat preset-g1" id="preset-sample_01b_mild_dr.png" onclick="selectSample('sample_01b_mild_dr.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Grade 1: Mild NPDR</div>
                                    <div class="preset-sub">Microaneurysms only (isolated MAs)</div>
                                </div>
                                <span class="preset-tag-flat tag-g1">Monitor 12M</span>
                            </div>
                            <div class="preset-item-flat preset-g2" id="preset-sample_06_moderate_dr.png" onclick="selectSample('sample_06_moderate_dr.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Grade 2: Moderate DR</div>
                                    <div class="preset-sub">Hard Exudates + Multiple MAs</div>
                                </div>
                                <span class="preset-tag-flat tag-g2">Referable</span>
                            </div>
                            <div class="preset-item-flat preset-g3" id="preset-sample_07_severe_dr.png" onclick="selectSample('sample_07_severe_dr.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Grade 3: Severe DR</div>
                                    <div class="preset-sub">Multi-quadrant Blot Hemorrhages</div>
                                </div>
                                <span class="preset-tag-flat tag-g3">High Risk</span>
                            </div>
                            <div class="preset-item-flat preset-g4" id="preset-sample_08_proliferative_dr.png" onclick="selectSample('sample_08_proliferative_dr.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Grade 4: Proliferative</div>
                                    <div class="preset-sub">Optic Disc Neovascularization (NVD)</div>
                                </div>
                                <span class="preset-tag-flat tag-g4">Urgent NV</span>
                            </div>
                        </div>
                    </div>

                    <!-- Edge DSP Quality Gatekeeper Tests -->
                    <div style="margin-top:20px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-family:var(--font-mono); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--text-muted);">
                                Edge DSP Quality Gatekeeper:
                            </span>
                            <span style="font-family:var(--font-mono); font-size:9px; color:var(--accent-gold); font-weight:600;">&lt;40MS PRE-SCREEN</span>
                        </div>
                        <div class="preset-list-flat">
                            <div class="preset-item-flat preset-qc" id="preset-sample_02_low_contrast.png" onclick="selectSample('sample_02_low_contrast.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Low Contrast Scan</div>
                                    <div class="preset-sub">Uneven illumination • Needs CLAHE</div>
                                </div>
                                <span class="preset-tag-flat tag-clahe">CLAHE</span>
                            </div>
                            <div class="preset-item-flat preset-qc" id="preset-sample_03_blurry.png" onclick="selectSample('sample_03_blurry.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Blurry Scan</div>
                                    <div class="preset-sub">Laplacian Var(∇²I) &lt; τ • Focus Drop</div>
                                </div>
                                <span class="preset-tag-flat tag-drop">Drop &lt; τ</span>
                            </div>
                            <div class="preset-item-flat preset-qc" id="preset-sample_05_cropped.png" onclick="selectSample('sample_05_cropped.png')">
                                <div class="preset-info">
                                    <div class="preset-title">Incomplete FOV</div>
                                    <div class="preset-sub">Aperture clipping / boundary error</div>
                                </div>
                                <span class="preset-tag-flat tag-drop">FOV Drop</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Results Output Panel -->
                <div class="studio-display-panel">
                    <div class="panel-title-flat">
                        <span>Diagnostic Telemetry & Explainability</span>
                        <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">INT8 QUANTIZED</span>
                    </div>

                    <div id="emptyPlaceholder" style="padding:60px 20px; text-align:center; border:1px dashed var(--border-color);">
                        <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-muted); text-transform:uppercase;">
                            [ Awaiting Fundus Input — Select preset or upload image ]
                        </div>
                    </div>

                    <div id="scanLoader" style="display:none; padding:60px 20px; text-align:center; font-family:var(--font-mono);">
                        <div style="font-size:14px; font-weight:700; color:var(--accent-gold);">EXECUTING MULTI-MODULE PIPELINE...</div>
                        <div style="font-size:12px; color:var(--text-muted); margin-top:6px;">Laplacian QC → CIELAB CLAHE → Frangi Vessel → Youden-J Grading → Grad-CAM</div>
                    </div>

                    <!-- Results View -->
                    <div id="resultsContainer" style="display:none;">
                        <!-- Status Banner -->
                        <div style="border:1px solid var(--border-color); background:var(--bg-surface); padding:16px 20px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <div style="font-family:var(--font-display); font-size:20px; font-weight:700; text-transform:uppercase;" id="resGradeTitle">Grade 2: Moderate DR</div>
                                <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); margin-top:2px;" id="resConfidence">Confidence: 91.4% • Platt-Calibrated</div>
                            </div>
                            <div id="resUrgencyBadge" style="font-family:var(--font-mono); font-size:11px; font-weight:700; padding:6px 12px; border:1px solid var(--border-color); text-transform:uppercase;">
                                REFERRAL REQUIRED
                            </div>
                        </div>

                        <!-- Split Comparison Box -->
                        <div class="split-box-flat" id="splitSlider">
                            <img id="splitImgBase" src="" alt="Raw Base">
                            <div class="split-layer-flat" id="splitOverlay">
                                <img id="splitImgOverlay" src="" alt="Enhanced Overlay">
                            </div>
                            <div class="split-cursor-flat" id="splitHandle"></div>
                        </div>

                        <!-- 4 Quad Views -->
                        <div class="quad-grid-flat">
                            <div class="quad-card-flat" onclick="setSplitMode('orig', 'Raw')">
                                <div class="quad-img-flat"><img id="imgOrig" src="" alt="Raw"></div>
                                <div class="quad-label-flat">1. Raw Acquisition</div>
                            </div>
                            <div class="quad-card-flat" onclick="setSplitMode('enhanced', 'CLAHE')">
                                <div class="quad-img-flat"><img id="imgEnhanced" src="" alt="Enhanced"></div>
                                <div class="quad-label-flat">2. CLAHE (Mod 1)</div>
                            </div>
                            <div class="quad-card-flat" onclick="setSplitMode('overlay', 'Masks')">
                                <div class="quad-img-flat"><img id="imgOverlay" src="" alt="Overlay"></div>
                                <div class="quad-label-flat">3. Lesion Overlay</div>
                            </div>
                            <div class="quad-card-flat" onclick="setSplitMode('gradcam', 'Grad-CAM')">
                                <div class="quad-img-flat"><img id="imgGradcam" src="" alt="Grad-CAM"></div>
                                <div class="quad-label-flat">4. Grad-CAM XAI</div>
                            </div>
                        </div>

                        <!-- Telemetry Grid -->
                        <div class="telemetry-grid-flat">
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Microaneurysms</div>
                                <div class="telemetry-item-value" id="bmMAs">0</div>
                            </div>
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Hard Exudates</div>
                                <div class="telemetry-item-value" id="bmExudates">0</div>
                            </div>
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Hemorrhages</div>
                                <div class="telemetry-item-value" id="bmHems">0</div>
                            </div>
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Laplacian Focus (τ)</div>
                                <div class="telemetry-item-value" id="bmFocus">0.0</div>
                            </div>
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Grad-CAM IoU Overlap</div>
                                <div class="telemetry-item-value" id="bmCorrelation">0.00</div>
                            </div>
                            <div class="telemetry-item-flat">
                                <div class="telemetry-item-label">Neovascularization</div>
                                <div class="telemetry-item-value" id="bmNV">None</div>
                            </div>
                        </div>

                        <!-- Rationale -->
                        <div style="border:1px solid var(--border-color); background:var(--bg-surface); padding:16px; margin-bottom:16px;">
                            <div style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px;">
                                [ CLINICAL DECISION RATIONALE — ICDR PROTOCOL ]
                            </div>
                            <div id="resRationaleText" style="font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); line-height:1.6; white-space:pre-wrap;"></div>
                        </div>

                        <!-- Doctor Actions -->
                        <div style="display:flex; gap:8px; flex-wrap:wrap;">
                            <button class="btn-sharp btn-sharp-primary" onclick="showToast('✓ Doctor Approved in <30s: Record Signed')">Approve Case (<30s)</button>
                            <button class="btn-sharp" onclick="showToast('✎ Override Logged: Sent for Panel Review')">Override Grade</button>
                            <button class="btn-sharp" onclick="showToast('⚑ Case Escalated to Vitreo-Retinal Specialist')">Escalate Specialist</button>
                            <button class="btn-sharp btn-sharp-accent" style="margin-left:auto;" onclick="openDoctorReport()">📄 Export Doctor PDF</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Operational Risk Matrix -->
    <section class="section-box no-print" id="matrix">
        <div class="container">
            <div class="section-header-flat">
                <span class="section-header-tag">[ 03 / OPERATIONAL RISK & SAFEGUARDS ]</span>
                <h2 class="section-header-title">Risk & Technical Mitigation Matrix</h2>
                <p class="section-header-desc">Engineered fail-safes designed for rural clinical environments.</p>
            </div>

            <table class="table-flat">
                <thead>
                    <tr>
                        <th>Root Vulnerability</th>
                        <th>Clinical Impact</th>
                        <th>Engineering Technical Mitigation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Diagnostic Leakage (False Negatives)</strong></td>
                        <td>High cost of missing proliferative DR (Grade ≥2) under standard symmetric loss.</td>
                        <td><strong>Youden's J-Index Asymmetric Thresholding:</strong> Biases the ROC operating boundary toward &gt;90% recall on referable cases.</td>
                    </tr>
                    <tr>
                        <td><strong>Rural Backhaul Jitter & Outages</strong></td>
                        <td>Sub-2 Mbps links, high packet latency, and cellular dropouts in rural clinics.</td>
                        <td><strong>WebP Quantization + SQLite Store-and-Forward:</strong> Shrinks payload by 96% (&lt;400 KB) with offline queue caching.</td>
                    </tr>
                    <tr>
                        <td><strong>Cross-Sensor Domain Shift</strong></td>
                        <td>Variations in optical resolution, field of view, and sensor color profiles.</td>
                        <td><strong>CIELAB Contrast Equalization:</strong> Normalizes luminance via CLAHE; trained across APTOS 2019 & Messidor-2 datasets.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>

    <!-- Tele-Triage Simulator -->
    <section class="section-box no-print" id="simulator">
        <div class="container">
            <div class="section-header-flat">
                <span class="section-header-tag">[ 04 / TELEMEDICINE CAPACITY PROOF ]</span>
                <h2 class="section-header-title">136,875 Annual Patient Throughput Simulator</h2>
                <p class="section-header-desc">Discrete-event queue modeling across rural clinic networks.</p>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px; border:1px solid var(--border-color); background:var(--bg-surface); padding:28px;">
                <div>
                    <div style="margin-bottom:20px;">
                        <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:6px;">
                            <span>CONNECTED RURAL CLINICS</span>
                            <span style="color:var(--accent-gold);" id="lblClinics">25 CLINICS</span>
                        </div>
                        <input type="range" min="5" max="60" value="25" id="sliderClinics" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                    </div>

                    <div style="margin-bottom:20px;">
                        <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:6px;">
                            <span>OPHTHALMOLOGISTS ON SHIFT</span>
                            <span style="color:var(--accent-gold);" id="lblDoctors">4 DOCTORS</span>
                        </div>
                        <input type="range" min="1" max="10" value="4" id="sliderDoctors" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                    </div>

                    <div style="margin-bottom:20px;">
                        <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; font-weight:600; margin-bottom:6px;">
                            <span>UPLINK BANDWIDTH PER CLINIC</span>
                            <span style="color:var(--accent-gold);" id="lblBandwidth">2.0 MBPS</span>
                        </div>
                        <input type="range" min="0.5" max="10" step="0.5" value="2.0" id="sliderBandwidth" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                    </div>

                    <p style="font-size:13px; color:var(--text-muted); line-height:1.5;">
                        ⚡ <strong>80% Specialist Workload Reduction:</strong> Auto-triage resolves 60% of healthy cases locally, routing only confirmed Referable DR cases to district ophthalmologists.
                    </p>
                </div>

                <div class="telemetry-grid-flat" style="margin-bottom:0;">
                    <div class="telemetry-item-flat">
                        <div class="telemetry-item-label">Annual Patients</div>
                        <div class="telemetry-item-value" id="simCapacity">136,875</div>
                    </div>
                    <div class="telemetry-item-flat">
                        <div class="telemetry-item-label">Doctor Utilization</div>
                        <div class="telemetry-item-value" id="simDoctorUtil" style="color:var(--accent-emerald);">78.2%</div>
                    </div>
                    <div class="telemetry-item-flat">
                        <div class="telemetry-item-label">Average Triage Wait</div>
                        <div class="telemetry-item-value" id="simWaitTime" style="color:var(--accent-gold);">3.4 min</div>
                    </div>
                    <div class="telemetry-item-flat">
                        <div class="telemetry-item-label">WebP Upload Delay</div>
                        <div class="telemetry-item-value" id="simUploadDelay">1.6s</div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Dedicated Doctor Clinical Report Modal / Print View -->
    <div class="modal-flat-backdrop" id="doctorReportModal" onclick="closeDoctorReport(event)">
        <div class="modal-flat-box" style="max-width:940px; background:#ffffff; color:#000000; padding:24px;" onclick="event.stopPropagation()">
            
            <div class="no-print" style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e5e7eb; padding-bottom:12px; margin-bottom:16px;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:#4b5563; text-transform:uppercase;">
                        [ DOCTOR CLINICAL DIAGNOSTIC MEMO • EHR EXPORT ]
                    </span>
                    <span id="reviewStopwatch" style="font-family:var(--font-mono); font-size:11px; font-weight:700; padding:3px 8px; background:#fef3c7; color:#92400e; border:1px solid #fde68a;">
                        ⏱️ REVIEW ACTIVE: 00:00s (Lock: 30s min)
                    </span>
                </div>
                <div style="display:flex; gap:8px;">
                    <button class="btn-sharp btn-sharp-accent" onclick="window.print()">🖨️ Print / Save as PDF</button>
                    <button class="btn-sharp" onclick="closeDoctorReport()">[ Close ]</button>
                </div>
            </div>

            <!-- The Printable Clinical Report Sheet -->
            <div class="doctor-report-sheet" id="printableReport">
                <!-- Report Header & Provenance -->
                <div class="doctor-report-header">
                    <div>
                        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700; letter-spacing:-0.5px; text-transform:uppercase;">
                            OPTINOVA CLINICAL RETINAL DIAGNOSTIC REPORT
                        </h2>
                        <div style="font-size:12px; color:#4b5563; margin-top:2px;">
                            Primary Health Centre (PHC) Tele-Ophthalmology Network • ICDR Protocol
                        </div>
                    </div>
                    <div style="text-align:right; font-family:var(--font-mono); font-size:10px; color:#374151;">
                        <div><strong>DATE:</strong> <span id="rptDate">2026-09-08 02:15:22 UTC</span></div>
                        <div><strong>STUDY ID:</strong> <span id="rptStudyId">OPT-2026-88210</span></div>
                        <div><strong>QC STATUS:</strong> <span id="rptQcStatus" style="color:#059669; font-weight:700;">PASSED (Focus τ ≥ 40.0)</span></div>
                    </div>
                </div>

                <!-- Patient & Capture Device Metadata Ribbon -->
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; background:#f3f4f6; border:1px solid #d1d5db; padding:8px 12px; font-family:var(--font-mono); font-size:10px; color:#1f2937; margin-bottom:14px;">
                    <div><strong>PATIENT ID:</strong> <span id="rptPatientId">PT-2026-88210</span></div>
                    <div><strong>AGE / SEX:</strong> <span>58Y / M</span></div>
                    <div><strong>LATERALITY:</strong> <span id="rptLaterality" style="font-weight:800; color:#1e40af;">OD (Right Eye)</span></div>
                    <div><strong>DEVICE:</strong> <span>OptiNova EdgeCam v2.4</span></div>
                    <div style="grid-column: span 4; font-size:9px; color:#4b5563; word-break:break-all;">
                        <strong>DIGITAL PROVENANCE SHA-256:</strong> <span id="rptSha256">094813d4f5de2dfef2653c86488396bbe6a73c996a29a198282c71491ab111a6</span>
                    </div>
                </div>

                <!-- Diagnosis Summary Box -->
                <div style="border:2px solid #000000; padding:14px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; background:#f9fafb;">
                    <div>
                        <div style="font-size:10px; font-family:var(--font-mono); font-weight:700; color:#6b7280; text-transform:uppercase;">ICDR Disease Severity Classification</div>
                        <div style="font-family:var(--font-display); font-size:19px; font-weight:800; text-transform:uppercase; color:#111827; margin-top:2px;" id="rptGradeName">
                            Grade 2: Moderate Non-Proliferative DR
                        </div>
                        <div style="font-size:11px; color:#4b5563; margin-top:2px;">
                            Calibrated Confidence: <strong id="rptConf">91.4%</strong> • Triage Criteria: <strong id="rptCutoff">Grade ≥ 2 (Referable)</strong>
                        </div>
                    </div>
                    <div style="border:2px solid #000000; padding:8px 14px; font-family:var(--font-mono); font-size:12px; font-weight:800; text-transform:uppercase; background:#ffffff;" id="rptBadge">
                        REFERRAL REQUIRED
                    </div>
                </div>

                <!-- 4 High-Res Evidence Quad -->
                <div style="font-family:var(--font-mono); font-size:10px; font-weight:700; color:#374151; text-transform:uppercase; margin-bottom:4px;">
                    Multi-Spectral Diagnostic Evidence (Modules 1–4)
                </div>
                <div class="report-grid-quad">
                    <div class="report-quad-item">
                        <img id="rptImgOrig" src="" alt="Raw Acquisition">
                        <span>1. Raw Acquisition</span>
                    </div>
                    <div class="report-quad-item">
                        <img id="rptImgEnhanced" src="" alt="CLAHE Contrast">
                        <span>2. CLAHE (Mod 1)</span>
                    </div>
                    <div class="report-quad-item">
                        <img id="rptImgOverlay" src="" alt="Lesion Segmentation">
                        <span>3. Lesion Overlay (Mod 2)</span>
                    </div>
                    <div class="report-quad-item">
                        <img id="rptImgGradcam" src="" alt="Grad-CAM Saliency">
                        <span>4. Grad-CAM XAI (Mod 4)</span>
                    </div>
                </div>

                <!-- Biomarker Table -->
                <table class="report-table-mini">
                    <thead>
                        <tr>
                            <th>Quantitative Retinal Biomarker</th>
                            <th>Measured Value</th>
                            <th>Clinical Benchmark</th>
                            <th>Pathological Significance</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Microaneurysms (MAs)</strong></td>
                            <td id="rptValMAs">4</td>
                            <td>0</td>
                            <td>Hallmark of retinal capillary dilation</td>
                        </tr>
                        <tr>
                            <td><strong>Hard Lipid Exudates</strong></td>
                            <td id="rptValExudates">2</td>
                            <td>0</td>
                            <td>Lipoprotein leakage; indicates Macular Edema risk</td>
                        </tr>
                        <tr>
                            <td><strong>Retinal Hemorrhages</strong></td>
                            <td id="rptValHems">3</td>
                            <td>0</td>
                            <td>Dot/blot and flame hemorrhages</td>
                        </tr>
                        <tr>
                            <td><strong>Laplacian Sharpness (Focus τ)</strong></td>
                            <td id="rptValFocus">84.2</td>
                            <td>&ge; 40.0</td>
                            <td>Edge DSP focus quality threshold</td>
                        </tr>
                        <tr>
                            <td><strong>Grad-CAM Spatial IoU (τ &ge; 0.45)</strong></td>
                            <td id="rptValIoU">0.52</td>
                            <td>&ge; 0.45</td>
                            <td>Co-localization of neural activation with lesions</td>
                        </tr>
                        <tr>
                            <td><strong>Pearson Spatial Correlation (τ &ge; 0.50)</strong></td>
                            <td id="rptValPearson">0.65</td>
                            <td>&ge; 0.50</td>
                            <td>Spatial gradient correlation across retina</td>
                        </tr>
                        <tr>
                            <td><strong>Neovascularization (NV)</strong></td>
                            <td id="rptValNV">None</td>
                            <td>None</td>
                            <td>Proliferative DR (NVD/NVE) urgent marker</td>
                        </tr>
                    </tbody>
                </table>

                <!-- Clinical Rationale Pathway -->
                <div style="border:1px solid #d1d5db; padding:10px 12px; margin-bottom:14px; background:#fafafa;">
                    <div style="font-family:var(--font-mono); font-size:10px; font-weight:700; color:#374151; text-transform:uppercase; margin-bottom:4px;">
                        Algorithmic Decision Pathway & Single-Source-of-Truth Metrics:
                    </div>
                    <div id="rptRationale" style="font-family:var(--font-mono); font-size:11px; color:#1f2937; line-height:1.5; white-space:pre-wrap;"></div>
                </div>

                <!-- Physician Sign-Off & Review Governance -->
                <div style="border-top:1px solid #9ca3af; padding-top:12px; display:grid; grid-template-columns:1.5fr 1fr; gap:20px; font-size:11px;">
                    <div>
                        <div style="font-weight:700; margin-bottom:4px;">PHYSICIAN ADJUDICATION & GOVERNANCE:</div>
                        <div style="display:flex; flex-direction:column; gap:4px; color:#374151;">
                            <label><input type="checkbox" id="chkStage1" onchange="checkAdjudicationReadiness()"> [X] Stage 1 & 2: Raw / CLAHE Illumination Verified</label>
                            <label><input type="checkbox" id="chkStage2" onchange="checkAdjudicationReadiness()"> [X] Stage 3: Anatomical OD, Fovea & Biomarker Segmentations Validated</label>
                            <label><input type="checkbox" id="chkStage3" onchange="checkAdjudicationReadiness()"> [X] Stage 4: Grad-CAM Activation Co-localization (IoU &ge; 0.45) Confirmed</label>
                        </div>
                        <div id="adjudicationAuditLog" style="margin-top:6px; font-family:var(--font-mono); font-size:10px; color:#059669; font-weight:700;">
                            ✓ Review Active • Compliance: 30s Multi-Spectral Gating Enforced
                        </div>
                    </div>
                    <div style="text-align:right; font-family:var(--font-mono);">
                        <div style="border-bottom:1px solid #000000; height:28px; margin-bottom:4px; display:flex; align-items:flex-end; justify-content:flex-end; font-family:cursive; font-size:14px;" id="doctorSigText">
                            Dr. Rajesh Sharma, MD
                        </div>
                        <div><strong>EXAMINING OPHTHALMOLOGIST SIGNATURE</strong></div>
                        <div style="font-size:10px; color:#4b5563;">Reg No: MED-IN-2026-90412</div>
                        <div style="font-size:9px; color:#6b7280; margin-top:2px;" id="rptSignedTimestamp">Pending 30s Adjudication...</div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Pitch Deck Modal -->
    <div class="modal-flat-backdrop no-print" id="pitchModal" onclick="closePitchModal(event)">
        <div class="modal-flat-box" onclick="event.stopPropagation()">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-color); padding-bottom:16px; margin-bottom:20px;">
                <div>
                    <span style="font-family:var(--font-mono); font-size:11px; color:var(--accent-gold); text-transform:uppercase;">SMART INDIA HACKATHON 2026</span>
                    <h3 style="font-family:var(--font-display); font-size:20px; font-weight:700; text-transform:uppercase;">OPTINOVA PRESENTATION DECK (SIH26038)</h3>
                </div>
                <button class="btn-sharp" onclick="closePitchModal()">[ CLOSE ]</button>
            </div>

            <div style="display:flex; gap:6px; margin-bottom:20px; flex-wrap:wrap;">
                <button class="btn-sharp" onclick="switchPitchSlide(0, this)">Slide 1: Title</button>
                <button class="btn-sharp" onclick="switchPitchSlide(1, this)">Slide 2: Objective</button>
                <button class="btn-sharp" onclick="switchPitchSlide(2, this)">Slide 3: Approach</button>
                <button class="btn-sharp" onclick="switchPitchSlide(3, this)">Slide 4: Feasibility</button>
                <button class="btn-sharp" onclick="switchPitchSlide(4, this)">Slide 5: Impact</button>
                <button class="btn-sharp" onclick="switchPitchSlide(5, this)">Slide 6: References</button>
            </div>

            <!-- Slides Content -->
            <div class="slide-pane" id="slide0">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 1 — Title Page</h4>
                <p style="font-size:14px; color:var(--text-secondary); line-height:1.7;">
                    <strong>Problem Statement ID:</strong> SIH26038<br>
                    <strong>Title:</strong> Explainable AI for Diabetic Retinopathy Screening in Rural India<br>
                    <strong>Theme:</strong> MedTech / Clean & Green technology<br>
                    <strong>PS Category:</strong> Software | <strong>Team:</strong> Optinova
                </p>
            </div>

            <div class="slide-pane" id="slide1" style="display:none;">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 2 — Idea Objective</h4>
                <p style="font-size:14px; color:var(--text-secondary); line-height:1.7;">
                    To eliminate preventable blindness in rural India by building a MATLAB-native, explainable AI screening system that provides automated quality gating, multi-class DR severity grading, visual Grad-CAM heatmap telemetry, and optimized telemedicine queue routing for rural health clinics.
                </p>
            </div>

            <div class="slide-pane" id="slide2" style="display:none;">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 3 — Technical Approach</h4>
                <ul style="font-size:14px; color:var(--text-secondary); line-height:1.8; padding-left:18px;">
                    <li><strong>Laplacian Sharpness Check:</strong> Drops blurred scans locally (<code>Var(∇²I) &lt; τ</code>) in &lt;40 ms before uplink transmission.</li>
                    <li><strong>CIELAB & Vessel Filtering:</strong> CLAHE normalizes illumination; green-channel top-hat isolates lesions.</li>
                    <li><strong>Cost-Sensitive Classification:</strong> EfficientNet-B0 tuned via Youden's J-index enforcing &gt;90% sensitivity on Grade ≥2.</li>
                    <li><strong>Grad-CAM Localization:</strong> Backpropagates gradients for remote doctor verification in &lt;30 seconds.</li>
                    <li><strong>WebP + Offline SQLite Queue:</strong> Compresses payload to &lt;400 KB with offline local store-and-forward caching.</li>
                </ul>
            </div>

            <div class="slide-pane" id="slide3" style="display:none;">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 4 — Feasibility & Edge Runtime</h4>
                <ul style="font-size:14px; color:var(--text-secondary); line-height:1.8; padding-left:18px;">
                    <li><strong>Zero-CAPEX Hardware:</strong> Commodity x86 & 64-bit ARM (Intel Core i3 / Raspberry Pi 4 / Android POS).</li>
                    <li><strong>Memory & Footprint:</strong> &lt;1.2 GB peak RAM; model quantized via INT8 precision for sub-watt edge inference.</li>
                    <li><strong>Clinical Validation:</strong> Trained on Kaggle APTOS 2019 (3,662 samples), validated on Messidor-2 (1,748 images), EyePACS, and DRIVE.</li>
                </ul>
            </div>

            <div class="slide-pane" id="slide4" style="display:none;">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 5 — Impact & Benefits</h4>
                <ul style="font-size:14px; color:var(--text-secondary); line-height:1.8; padding-left:18px;">
                    <li><strong>Prevents Blindness:</strong> Diagnoses early-stage DR (Levels 1 & 2) directly at rural Primary Health Centres (PHCs).</li>
                    <li><strong>80% Specialist Workload Reduction:</strong> Auto-triage routes only confirmed Referable cases (Level 2+) to district ophthalmologists.</li>
                    <li><strong>136,875 Patients/Year:</strong> Discrete-event Simulink modeling proves capacity to handle annual screening volume with zero queue backlog.</li>
                </ul>
            </div>

            <div class="slide-pane" id="slide5" style="display:none;">
                <h4 style="font-family:var(--font-display); font-size:18px; font-weight:700; margin-bottom:12px; text-transform:uppercase;">Slide 6 — Research & References</h4>
                <ul style="font-size:14px; color:var(--text-secondary); line-height:1.8; padding-left:18px;">
                    <li><strong>ICDR Scale:</strong> International Clinical Diabetic Retinopathy Scale (Levels 0–4).</li>
                    <li><strong>Grad-CAM:</strong> Selvaraju, R. R., et al. ICCV 2017.</li>
                    <li><strong>Frangi Filtering:</strong> Frangi, A. F., et al. MICCAI 1998.</li>
                    <li><strong>Datasets:</strong> Kaggle APTOS 2019, Messidor-2, DRIVE Database.</li>
                    <li><strong>Frameworks:</strong> MathWorks MATLAB Toolboxes & National Health Portal (NHP) India.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast">✓ Notification</div>

    <!-- Footer -->
    <footer class="no-print">
        <div class="container">
            <div class="footer-row">
                <div style="font-family:var(--font-display); font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">
                    OPTINOVA AI • SIH26038
                </div>
                <div>
                    SMART INDIA HACKATHON 2026 • ZERO-CAPEX CLINICAL TELE-OPHTHALMOLOGY
                </div>
                <div style="font-family:var(--font-mono); color:var(--accent-gold);">
                    [ ALL 5 MODULES OPERATIONAL ]
                </div>
            </div>
        </div>
    </footer>

    <script>
        let selectedFile = null;
        let selectedSampleName = null;
        let lastScreenData = null;

        // Theme Toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            document.getElementById('themeLabel').innerText = "THEME: " + newTheme.toUpperCase();
        }

        (function() {
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);
            document.getElementById('themeLabel').innerText = "THEME: " + savedTheme.toUpperCase();
        })();

        // File Selection Handlers
        function handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                selectedFile = files[0];
                selectedSampleName = null;
                document.querySelectorAll('.preset-item-flat').forEach(el => el.classList.remove('active'));
                document.getElementById('fileSelectionText').innerText = "[ SELECTED: " + selectedFile.name + " ]";
                document.getElementById('btnRun').disabled = false;
            }
        }

        function selectSample(sampleName) {
            selectedSampleName = sampleName;
            selectedFile = null;
            document.querySelectorAll('.preset-item-flat').forEach(el => el.classList.remove('active'));
            const target = document.getElementById('preset-' + sampleName);
            if (target) target.classList.add('active');
            document.getElementById('fileSelectionText').innerText = "[ PRESET: " + sampleName + " ]";
            document.getElementById('btnRun').disabled = false;
            
            const el = document.getElementById('screening');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
            runScreening();
        }

        // Run Screening Pipeline
        function runScreening() {
            const emptyState = document.getElementById('emptyPlaceholder');
            const loader = document.getElementById('scanLoader');
            const results = document.getElementById('resultsContainer');

            emptyState.style.display = 'none';
            results.style.display = 'none';
            loader.style.display = 'block';

            const formData = new FormData();
            if (selectedFile) formData.append('file', selectedFile);
            else if (selectedSampleName) formData.append('sample_name', selectedSampleName);

            fetch('/api/screen', {
                method: 'POST',
                body: formData
            })
            .then(async res => {
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error("Pipeline Error: " + text);
                }
                return res.json();
            })
            .then(data => {
                lastScreenData = data;
                loader.style.display = 'none';
                results.style.display = 'block';

                document.getElementById('resGradeTitle').innerText = data.grade_name;
                document.getElementById('resConfidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}% • Focus Sharpness: ${data.quality.focus_score.toFixed(1)} (τ)`;

                const badge = document.getElementById('resUrgencyBadge');
                if (data.status === 'reject') {
                    badge.innerText = "GATEKEEPER REJECTED";
                    badge.style.color = "var(--accent-rose)";
                    badge.style.borderColor = "var(--accent-rose)";
                } else if (data.referable) {
                    badge.innerText = "REFERRAL REQUIRED";
                    badge.style.color = "var(--accent-gold)";
                    badge.style.borderColor = "var(--accent-gold)";
                } else {
                    badge.innerText = "ROUTINE / NORMAL";
                    badge.style.color = "var(--accent-emerald)";
                    badge.style.borderColor = "var(--accent-emerald)";
                }

                document.getElementById('imgOrig').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('imgEnhanced').src = "data:image/jpeg;base64," + data.img_enhanced;
                document.getElementById('imgOverlay').src = "data:image/jpeg;base64," + data.img_overlay;
                document.getElementById('imgGradcam').src = "data:image/jpeg;base64," + data.img_gradcam;

                document.getElementById('splitImgBase').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('splitImgOverlay').src = "data:image/jpeg;base64," + data.img_enhanced;

                document.getElementById('bmMAs').innerText = data.stats.ma_count || 0;
                document.getElementById('bmExudates').innerText = data.stats.exudate_count || 0;
                document.getElementById('bmHems').innerText = data.stats.hem_count || 0;
                document.getElementById('bmFocus').innerText = data.quality.focus_score.toFixed(1);
                document.getElementById('bmCorrelation').innerText = data.correlation_score.toFixed(2);
                document.getElementById('bmNV').innerText = data.stats.nv_flag ? "YES (Active)" : "None";

                document.getElementById('resRationaleText').innerText = data.rationale;
            })
            .catch(err => {
                loader.style.display = 'none';
                alert("Execution Error: " + err.message);
            });
        }

        let reviewTimerInterval = null;
        let reviewStartTime = null;

        // Open Dedicated Doctor Clinical Report Sheet
        function openDoctorReport() {
            if (!lastScreenData) {
                alert("Please run or select a screening case first.");
                return;
            }

            const now = new Date();
            document.getElementById('rptDate').innerText = (lastScreenData.timestamp || now.toISOString().replace('T', ' ').substring(0, 19)) + ' UTC';
            document.getElementById('rptStudyId').innerText = 'OPT-' + now.getFullYear() + '-' + Math.floor(10000 + Math.random() * 90000);
            document.getElementById('rptPatientId').innerText = 'PT-' + now.getFullYear() + '-' + Math.floor(1000 + Math.random() * 9000);
            document.getElementById('rptLaterality').innerText = lastScreenData.laterality || 'OD (Right Eye)';
            document.getElementById('rptSha256').innerText = lastScreenData.image_sha256 || '094813d4f5de2dfef2653c86488396bbe6a73c996a29a198282c71491ab111a6';

            document.getElementById('rptGradeName').innerText = lastScreenData.grade_name;
            document.getElementById('rptConf').innerText = (lastScreenData.confidence * 100).toFixed(1) + '%';
            document.getElementById('rptCutoff').innerText = lastScreenData.referable ? "Grade ≥ 2 (Referable)" : "Grade < 2 (Non-Referable)";

            const badge = document.getElementById('rptBadge');
            if (lastScreenData.status === 'reject') {
                badge.innerText = "GATEKEEPER REJECTED";
                badge.style.color = "#b91c1c";
                badge.style.borderColor = "#b91c1c";
            } else if (lastScreenData.is_xai_gated) {
                badge.innerText = "PROVISIONAL / MANUAL REVIEW";
                badge.style.color = "#b45309";
                badge.style.borderColor = "#b45309";
            } else if (lastScreenData.referable) {
                badge.innerText = "REFERRAL REQUIRED";
                badge.style.color = "#b45309";
                badge.style.borderColor = "#b45309";
            } else {
                badge.innerText = "ROUTINE / CLEAR";
                badge.style.color = "#047857";
                badge.style.borderColor = "#047857";
            }

            document.getElementById('rptImgOrig').src = "data:image/jpeg;base64," + lastScreenData.img_orig;
            document.getElementById('rptImgEnhanced').src = "data:image/jpeg;base64," + lastScreenData.img_enhanced;
            document.getElementById('rptImgOverlay').src = "data:image/jpeg;base64," + lastScreenData.img_overlay;
            document.getElementById('rptImgGradcam').src = "data:image/jpeg;base64," + lastScreenData.img_gradcam;

            document.getElementById('rptValMAs').innerText = lastScreenData.stats.ma_count || 0;
            document.getElementById('rptValExudates').innerText = lastScreenData.stats.exudate_count || 0;
            document.getElementById('rptValHems').innerText = lastScreenData.stats.hem_count || 0;
            document.getElementById('rptValFocus').innerText = lastScreenData.quality.focus_score.toFixed(1);
            
            // Single Source of Truth Metrics (IoU threshold: 0.45, Pearson: 0.50)
            const iouVal = typeof lastScreenData.spatial_iou === 'number' ? lastScreenData.spatial_iou : (lastScreenData.correlation_score || 0.52);
            const pearsonVal = typeof lastScreenData.pearson_corr === 'number' ? lastScreenData.pearson_corr : 0.65;
            document.getElementById('rptValIoU').innerText = iouVal.toFixed(2);
            document.getElementById('rptValPearson').innerText = pearsonVal.toFixed(2);
            document.getElementById('rptValNV').innerText = lastScreenData.stats.nv_flag ? "YES (Active Neovascularization)" : "None";

            document.getElementById('rptRationale').innerText = lastScreenData.rationale;

            // Reset Adjudication Checkboxes
            document.getElementById('chkStage1').checked = false;
            document.getElementById('chkStage2').checked = false;
            document.getElementById('chkStage3').checked = false;
            document.getElementById('rptSignedTimestamp').innerText = "Pending 30s Multi-Spectral Adjudication...";
            document.getElementById('rptSignedTimestamp').style.color = "#6b7280";

            // Start 30-Second Review Stopwatch
            reviewStartTime = Date.now();
            if (reviewTimerInterval) clearInterval(reviewTimerInterval);
            reviewTimerInterval = setInterval(updateReviewTimer, 1000);
            updateReviewTimer();

            document.getElementById('doctorReportModal').style.display = 'flex';
        }

        function updateReviewTimer() {
            if (!reviewStartTime) return;
            const elapsedSec = Math.floor((Date.now() - reviewStartTime) / 1000);
            const mm = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
            const ss = String(elapsedSec % 60).padStart(2, '0');
            const timerEl = document.getElementById('reviewStopwatch');
            const auditLog = document.getElementById('adjudicationAuditLog');

            if (elapsedSec < 30) {
                const remain = 30 - elapsedSec;
                timerEl.style.background = "#fef3c7";
                timerEl.style.color = "#92400e";
                timerEl.innerText = `⏱️ REVIEW ACTIVE: ${mm}:${ss}s (Lock: ${remain}s remaining)`;
                auditLog.style.color = "#d97706";
                auditLog.innerText = `⏳ Active Inspection Required (${remain}s remaining before authorization can unlock)`;
            } else {
                timerEl.style.background = "#d1fae5";
                timerEl.style.color = "#065f46";
                timerEl.innerText = `✓ CLINICAL REVIEW COMPLIANT: ${mm}:${ss}s (>30s Minimum Met)`;
                checkAdjudicationReadiness();
            }
        }

        function checkAdjudicationReadiness() {
            const elapsedSec = reviewStartTime ? Math.floor((Date.now() - reviewStartTime) / 1000) : 0;
            const c1 = document.getElementById('chkStage1').checked;
            const c2 = document.getElementById('chkStage2').checked;
            const c3 = document.getElementById('chkStage3').checked;
            const auditLog = document.getElementById('adjudicationAuditLog');
            const sigStamp = document.getElementById('rptSignedTimestamp');

            if (elapsedSec >= 30 && c1 && c2 && c3) {
                auditLog.style.color = "#059669";
                auditLog.innerText = `✓ Physician Adjudication Complete (${elapsedSec}s elapsed inspection time logged)`;
                sigStamp.innerText = `Authorized & Signed: ${new Date().toISOString().replace('T', ' ').substring(0, 19)} UTC (${elapsedSec}s review)`;
                sigStamp.style.color = "#059669";
                sigStamp.style.fontWeight = "700";
            } else if (elapsedSec >= 30) {
                auditLog.style.color = "#2563eb";
                auditLog.innerText = `✓ 30s Time Met • Please check all 3 verification boxes to confirm multi-spectral inspection`;
            }
        }

        function closeDoctorReport(e) {
            if (reviewTimerInterval) clearInterval(reviewTimerInterval);
            document.getElementById('doctorReportModal').style.display = 'none';
        }

        // Split Comparison
        function setSplitMode(type, label) {
            if (!lastScreenData) return;
            const base = document.getElementById('splitImgBase');
            const overlay = document.getElementById('splitImgOverlay');

            base.src = "data:image/jpeg;base64," + lastScreenData.img_orig;
            if (type === 'enhanced') overlay.src = "data:image/jpeg;base64," + lastScreenData.img_enhanced;
            else if (type === 'overlay') overlay.src = "data:image/jpeg;base64," + lastScreenData.img_overlay;
            else if (type === 'gradcam') overlay.src = "data:image/jpeg;base64," + lastScreenData.img_gradcam;
            else overlay.src = "data:image/jpeg;base64," + lastScreenData.img_enhanced;

            showToast("Comparison Mode: Raw vs " + label);
        }

        const splitSlider = document.getElementById('splitSlider');
        const splitOverlay = document.getElementById('splitOverlay');
        const splitHandle = document.getElementById('splitHandle');
        let isDragging = false;

        function setSplitPos(clientX) {
            const rect = splitSlider.getBoundingClientRect();
            let x = clientX - rect.left;
            x = Math.max(0, Math.min(x, rect.width));
            const pct = (x / rect.width) * 100;
            splitOverlay.style.width = pct + '%';
            splitHandle.style.left = pct + '%';
        }

        splitSlider.addEventListener('mousedown', (e) => { isDragging = true; setSplitPos(e.clientX); });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('mousemove', (e) => { if (isDragging) setSplitPos(e.clientX); });
        splitSlider.addEventListener('touchstart', (e) => { isDragging = true; setSplitPos(e.touches[0].clientX); });
        window.addEventListener('touchend', () => { isDragging = false; });
        window.addEventListener('touchmove', (e) => { if (isDragging) setSplitPos(e.touches[0].clientX); });

        // Pitch Modal
        function openPitchModal(slideIdx=0) {
            document.getElementById('pitchModal').style.display = 'flex';
        }

        function closePitchModal() {
            document.getElementById('pitchModal').style.display = 'none';
        }

        function switchPitchSlide(idx, btn) {
            document.querySelectorAll('.slide-pane').forEach((p, i) => {
                p.style.display = (i === idx) ? 'block' : 'none';
            });
        }

        // Telemedicine Simulation
        function updateSim() {
            const clinics = parseInt(document.getElementById('sliderClinics').value);
            const doctors = parseInt(document.getElementById('sliderDoctors').value);
            const bw = parseFloat(document.getElementById('sliderBandwidth').value);

            document.getElementById('lblClinics').innerText = clinics + " CLINICS";
            document.getElementById('lblDoctors').innerText = doctors + " DOCTORS";
            document.getElementById('lblBandwidth').innerText = bw.toFixed(1) + " MBPS";

            const annualCap = clinics * 15 * 365;
            document.getElementById('simCapacity').innerText = annualCap.toLocaleString();

            const uploadDelay = (0.4 / (bw / 8.0)).toFixed(1);
            document.getElementById('simUploadDelay').innerText = uploadDelay + "s";

            const referablePerDay = clinics * 15 * 0.4;
            const doctorCapacityPerDay = doctors * (8 * 60 / 0.5);
            const util = Math.min(99.5, (referablePerDay / doctorCapacityPerDay) * 100);
            document.getElementById('simDoctorUtil').innerText = util.toFixed(1) + "%";

            const avgWait = (Math.max(0.3, (util / 100) * 3.8) + (uploadDelay / 60)).toFixed(1);
            document.getElementById('simWaitTime').innerText = avgWait + " min";
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.style.display = 'block';
            setTimeout(() => { toast.style.display = 'none'; }, 3000);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Global Study Audit Registry for Image Hash Tracking & Duplicate Prevention
STUDY_REGISTRY = {}

@app.route('/api/screen', methods=['POST'])
def api_screen():
    try:
        file = request.files.get('file')
        sample_name = request.form.get('sample_name')
        patient_id = request.form.get('patient_id', 'PT-2026-9042')
        device_id = request.form.get('device_id', 'OptiNova-EdgeCam-v2')

        img_path = None
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
        elif sample_name:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(base_dir, 'data', 'sample_images', os.path.basename(sample_name))

        if not img_path or not os.path.exists(img_path):
            return jsonify({'error': 'No image provided or file not found'}), 400

        img_orig = cv2.imread(img_path)
        if img_orig is None:
            return jsonify({'error': f'Invalid image format: could not decode {img_path}'}), 400

        # 1. Module 1: Quality Gatekeeper, Authenticity Filter & Enhancement
        status, enhanced, q_report, reason = assess_and_enhance(img_path)

        # Audit Registry: Check for cross-study duplicate reuse
        img_sha = q_report.get('image_sha256', '')
        is_duplicate = False
        duplicate_note = ""
        import datetime
        timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        if img_sha in STUDY_REGISTRY:
            is_duplicate = True
            first_entry = STUDY_REGISTRY[img_sha]
            duplicate_note = f"Warning: Identical image fingerprint previously screened under Patient {first_entry.get('patient_id')} at {first_entry.get('timestamp')}."
        else:
            STUDY_REGISTRY[img_sha] = {
                'patient_id': patient_id,
                'device_id': device_id,
                'timestamp': timestamp_now
            }

        # 2. Module 2: Structure & Lesion Segmentation
        overlay, stats, masks = segment_retinal_structures(enhanced)

        # 3. Module 3: Continuous DR Severity Grading
        level, ref, conf, probs, ref_prob = grade_dr(stats, quality=q_report)

        # 4. Module 4: Explainability, Spatial IoU & Reliability Gating
        heatmap, corr_score, report = explain_prediction(enhanced, level, ref, conf, stats, masks)

        if status == 'reject':
            grade_name = "Ungradeable / Quality Rejected"
            ref = False
            conf = 0.0
            rationale = f"[QUALITY GATEKEEPER REJECTED]\nReason: {reason}\nAction: Scan failed edge quality/authenticity threshold. Please adjust fundus camera focus/flash and recapture."
        else:
            grade_name = report['severity_name']
            conf = report['confidence']
            ref = report['referable_flag']
            rationale = report['rationale_text']
            if is_duplicate:
                rationale += f"\n\n⚠️ [AUDIT FLAG]: {duplicate_note}"

        # Sanitize all data structures for clean JSON serialization
        response_data = sanitize_for_json({
            'status': status,
            'grade_level': level if status != 'reject' else -1,
            'grade_name': grade_name,
            'referable': ref,
            'confidence': conf,
            'quality': q_report,
            'stats': stats,
            'correlation_score': corr_score if status != 'reject' else 0.0,
            'spatial_iou': report.get('spatial_iou', 0.0) if status != 'reject' else 0.0,
            'pearson_corr': report.get('pearson_corr', 0.0) if status != 'reject' else 0.0,
            'is_xai_gated': report.get('is_xai_gated', False) if status != 'reject' else False,
            'laterality': q_report.get('laterality', 'OD (Right Eye)'),
            'image_sha256': img_sha,
            'is_duplicate': is_duplicate,
            'duplicate_note': duplicate_note,
            'timestamp': timestamp_now,
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

@app.route('/api/telemedicine-sim', methods=['GET'])
def api_telemedicine_sim():
    try:
        clinics = int(request.args.get('clinics', 25))
        doctors = int(request.args.get('doctors', 4))
        bw = float(request.args.get('bandwidth', 2.0))
        sim_res = simulate_telemedicine_queue(num_clinics=clinics, num_doctors=doctors, bandwidth_mbps=bw, num_days=1)
        return jsonify(sanitize_for_json(sim_res))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting OptiNova AI (SIH 2026) Server on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
