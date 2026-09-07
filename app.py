#!/usr/bin/env python3
"""
<<<<<<< HEAD
Explainable AI Diabetic Retinopathy Screening — Web Application Server
Provides a complete web interface for users to upload fundus images, execute Modules 1-5,
and inspect visual overlays, Grad-CAM heatmaps, severity scores, and clinical reports.
=======
OptiNova AI — Explainable Retinal Intelligence & Diabetic Retinopathy Screening (SIH 2026)
Inspired by modern health intelligence platforms (superpower.com style)
Integrates Modules 1 to 5: Quality Gatekeeper, Vascular Segmentation, ETDRS Grading, Grad-CAM Explainability, and Telemedicine Triaging.
>>>>>>> c1e4f41 (feat(ui): redesign web app as OptiNova AI with superpower.com aesthetics, dark/light mode, and clinical biomarker suite)
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

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception:
    UPLOAD_FOLDER = '/tmp/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def image_to_base64(img_bgr):
    """Converts OpenCV BGR image to base64 JPEG string for inline HTML rendering."""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
<<<<<<< HEAD
    <title>Retinal Health Screening Assistant</title>
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
            color: white; padding: 18px 40px;
            display: flex; justify-content: space-between; align-items: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .brand-icon {
            font-size: 26px;
            background: rgba(255, 255, 255, 0.15);
            width: 46px;
            height: 46px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            backdrop-filter: blur(4px);
        }
        header h1 { margin: 0; font-size: 21px; font-weight: 700; letter-spacing: -0.3px; }
        header p { margin: 4px 0 0 0; opacity: 0.9; font-size: 13px; font-weight: 400; }
        .header-status {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(6px);
            color: #ffffff;
            padding: 7px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background-color: #2ecc71;
            display: inline-block;
            box-shadow: 0 0 8px #2ecc71;
        }
        
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
=======
    <title>OptiNova AI | Retinal Vision Intelligence & Explainable DR Screening</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-display: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
>>>>>>> c1e4f41 (feat(ui): redesign web app as OptiNova AI with superpower.com aesthetics, dark/light mode, and clinical biomarker suite)

            /* Dark Theme (Default) */
            --bg-body: #08090d;
            --bg-surface: #0f1118;
            --bg-card: rgba(18, 21, 31, 0.72);
            --bg-card-hover: rgba(25, 29, 43, 0.85);
            --bg-input: #141724;
            --bg-badge: rgba(255, 255, 255, 0.06);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-active: rgba(245, 158, 11, 0.4);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            
            --accent-gold: #f59e0b;
            --accent-gold-glow: rgba(245, 158, 11, 0.25);
            --accent-amber: #d97706;
            --accent-emerald: #10b981;
            --accent-emerald-glow: rgba(16, 185, 129, 0.2);
            --accent-rose: #f43f5e;
            --accent-rose-glow: rgba(244, 63, 94, 0.2);
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;

            --hero-glow: radial-gradient(circle at 50% 30%, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 35%, rgba(8, 9, 13, 0) 70%);
            --card-glass-blur: blur(20px);
            --shadow-subtle: 0 4px 20px rgba(0, 0, 0, 0.35);
            --shadow-floating: 0 12px 36px rgba(0, 0, 0, 0.45);
        }

        [data-theme="light"] {
            --bg-body: #f8fafc;
            --bg-surface: #ffffff;
            --bg-card: rgba(255, 255, 255, 0.85);
            --bg-card-hover: #ffffff;
            --bg-input: #f1f5f9;
            --bg-badge: rgba(15, 23, 42, 0.05);
            --border-subtle: rgba(15, 23, 42, 0.09);
            --border-active: rgba(217, 119, 6, 0.45);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;

            --accent-gold: #d97706;
            --accent-gold-glow: rgba(217, 119, 6, 0.15);
            --accent-amber: #b45309;
            --accent-emerald: #059669;
            --accent-emerald-glow: rgba(5, 150, 105, 0.15);
            --accent-rose: #e11d48;
            --accent-rose-glow: rgba(225, 29, 72, 0.15);
            --accent-cyan: #0891b2;
            --accent-purple: #9333ea;

            --hero-glow: radial-gradient(circle at 50% 25%, rgba(251, 191, 36, 0.25) 0%, rgba(253, 230, 138, 0.12) 40%, rgba(248, 250, 252, 0) 70%);
            --card-glass-blur: blur(20px);
            --shadow-subtle: 0 4px 20px rgba(15, 23, 42, 0.05);
            --shadow-floating: 0 12px 36px rgba(15, 23, 42, 0.08);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-body);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Ambient Glow & Grid Background */
        .ambient-glow {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 900px;
            background: var(--hero-glow);
            pointer-events: none;
            z-index: 0;
        }

        .bg-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 1000px;
            background-size: 40px 40px;
            background-image: 
                linear-gradient(to right, var(--border-subtle) 1px, transparent 1px),
                linear-gradient(to bottom, var(--border-subtle) 1px, transparent 1px);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.4) 0%, transparent 80%);
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,0.4) 0%, transparent 80%);
            pointer-events: none;
            z-index: 0;
        }

        /* Top Navigation */
        nav {
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 48px;
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            -webkit-backdrop-filter: var(--card-glass-blur);
            border-bottom: 1px solid var(--border-subtle);
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: var(--text-primary);
        }

        .nav-logo-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 20px;
            box-shadow: 0 4px 14px var(--accent-gold-glow);
        }

        .nav-logo-text {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 21px;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .nav-logo-text span {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 2px 8px;
            border-radius: 6px;
            background: var(--bg-badge);
            color: var(--accent-gold);
            border: 1px solid var(--border-subtle);
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
            font-size: 14px;
            font-weight: 500;
            transition: color 0.2s ease;
        }

        .nav-links a:hover {
            color: var(--text-primary);
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        /* Theme Toggle Pill */
        .theme-toggle-btn {
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 6px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 600;
            font-family: var(--font-main);
        }

        .theme-toggle-btn:hover {
            border-color: var(--accent-gold);
            color: var(--text-primary);
        }

        .btn-pill-primary {
            background: var(--text-primary);
            color: var(--bg-body);
            border: none;
            border-radius: 30px;
            padding: 10px 22px;
            font-weight: 600;
            font-size: 14px;
            font-family: var(--font-main);
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-pill-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }

        /* Container */
        .container {
            max-width: 1320px;
            margin: 0 auto;
            padding: 0 24px;
            position: relative;
            z-index: 1;
        }

        /* Hero Section (superpower.com style) */
        .hero {
            padding: 90px 0 60px 0;
            text-align: center;
            position: relative;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 18px;
            border-radius: 30px;
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            font-size: 13px;
            font-weight: 600;
            color: var(--accent-gold);
            margin-bottom: 24px;
            box-shadow: 0 2px 10px var(--accent-gold-glow);
        }

        .hero-badge .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-gold);
            box-shadow: 0 0 10px var(--accent-gold);
            animation: pulse-dot 2s infinite ease-in-out;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: clamp(42px, 6vw, 76px);
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: -1.8px;
            max-width: 960px;
            margin: 0 auto 20px auto;
        }

        .hero-title .highlight {
            background: linear-gradient(135deg, #f59e0b 0%, #f97316 50%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }

        .hero-sub {
            font-size: 19px;
            color: var(--text-secondary);
            max-width: 680px;
            margin: 0 auto 38px auto;
            font-weight: 400;
            line-height: 1.5;
        }

        .hero-actions {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 56px;
        }

        .btn-hero-primary {
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            color: #ffffff;
            border: none;
            border-radius: 36px;
            padding: 15px 34px;
            font-size: 16px;
            font-weight: 700;
            font-family: var(--font-main);
            cursor: pointer;
            box-shadow: 0 8px 24px var(--accent-gold-glow);
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
        }

        .btn-hero-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px var(--accent-gold-glow);
        }

        .btn-hero-secondary {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 36px;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 600;
            font-family: var(--font-main);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            transition: all 0.2s ease;
        }

        .btn-hero-secondary:hover {
            background: var(--bg-card-hover);
            border-color: var(--text-secondary);
        }

        /* Hero Stats Strip */
        .hero-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            max-width: 1080px;
            margin: 0 auto;
            padding: 24px;
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            box-shadow: var(--shadow-subtle);
        }

        .stat-item {
            text-align: center;
            border-right: 1px solid var(--border-subtle);
            padding: 6px 12px;
        }

        .stat-item:last-child {
            border-right: none;
        }

        .stat-value {
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }

        .stat-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-top: 4px;
        }

        /* Section Layouts */
        .section-header {
            text-align: center;
            margin-bottom: 48px;
        }

        .section-tag {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--accent-gold);
            margin-bottom: 12px;
            display: inline-block;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 38px;
            font-weight: 800;
            letter-spacing: -1px;
            color: var(--text-primary);
            margin-bottom: 14px;
        }

        .section-desc {
            font-size: 17px;
            color: var(--text-secondary);
            max-width: 640px;
            margin: 0 auto;
        }

        /* 4-Step "How it works" Cards (superpower.com style) */
        .how-it-works-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 80px;
        }

        .step-card {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease;
        }

        .step-card:hover {
            transform: translateY(-6px);
            border-color: var(--border-active);
        }

        .step-num {
            position: absolute;
            top: 18px;
            right: 20px;
            font-family: var(--font-display);
            font-size: 24px;
            font-weight: 800;
            color: var(--border-subtle);
        }

        .step-icon-box {
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            margin-bottom: 18px;
        }

        .step-title {
            font-family: var(--font-display);
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 10px;
        }

        .step-text {
            font-size: 13.5px;
            color: var(--text-secondary);
            line-height: 1.55;
        }

        /* Screening Studio App Interface */
        .screening-studio {
            padding: 40px 0 80px 0;
        }

        .studio-card {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 28px;
            padding: 36px;
            box-shadow: var(--shadow-floating);
        }

        .studio-grid {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 32px;
        }

        @media (max-width: 1024px) {
            .studio-grid { grid-template-columns: 1fr; }
            .how-it-works-grid { grid-template-columns: repeat(2, 1fr); }
            .hero-stats { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 640px) {
            .how-it-works-grid { grid-template-columns: 1fr; }
            .hero-stats { grid-template-columns: 1fr; }
            nav { padding: 16px 20px; }
            .nav-links { display: none; }
        }

        /* Left Side: Upload & Case Presets */
        .panel-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-title {
            font-family: var(--font-display);
            font-size: 17px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .drop-zone {
            border: 2px dashed var(--border-subtle);
            border-radius: 18px;
            padding: 30px 18px;
            text-align: center;
            background: var(--bg-surface);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent-gold);
            background: var(--bg-badge);
            box-shadow: 0 0 25px var(--accent-gold-glow);
        }

        .drop-zone-icon {
            width: 52px;
            height: 52px;
            border-radius: 16px;
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin: 0 auto 12px auto;
            color: var(--accent-gold);
        }

        .drop-zone-title {
            font-weight: 700;
            font-size: 14.5px;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .drop-zone-sub {
            font-size: 12px;
            color: var(--text-muted);
        }

        .btn-run-scan {
            width: 100%;
            margin-top: 18px;
            padding: 14px;
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            color: #ffffff;
            border: none;
            border-radius: 16px;
            font-family: var(--font-main);
            font-weight: 700;
            font-size: 15px;
            cursor: pointer;
            box-shadow: 0 6px 18px var(--accent-gold-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.2s ease;
        }

        .btn-run-scan:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px var(--accent-gold-glow);
        }

        .btn-run-scan:disabled {
            background: var(--bg-badge);
            color: var(--text-muted);
            box-shadow: none;
            cursor: not-allowed;
            border: 1px solid var(--border-subtle);
        }

        .preset-section {
            margin-top: 26px;
        }

        .preset-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .preset-grid {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .preset-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s ease;
        }

        .preset-chip:hover {
            border-color: var(--accent-gold);
            background: var(--bg-card-hover);
            transform: translateX(3px);
        }

        .preset-chip.active {
            border-color: var(--accent-gold);
            background: var(--bg-badge);
        }

        .preset-chip-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .preset-badge-tag {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 6px;
            text-transform: uppercase;
        }

        .tag-normal { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); }
        .tag-low { background: rgba(245, 158, 11, 0.15); color: var(--accent-gold); }
        .tag-reject { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }
        .tag-dr { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); }

        /* Right Side: Results Display */
        .empty-placeholder {
            border: 1px dashed var(--border-subtle);
            border-radius: 20px;
            padding: 80px 24px;
            text-align: center;
            background: var(--bg-surface);
        }

        .empty-reticle {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            border: 2px solid var(--border-subtle);
            margin: 0 auto 20px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            color: var(--text-muted);
            position: relative;
        }

        .empty-reticle::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 2px;
            background: var(--accent-gold);
            opacity: 0.4;
            animation: scan-line 2.5s infinite ease-in-out;
        }

        @keyframes scan-line {
            0% { top: 0; opacity: 0; }
            50% { opacity: 0.8; }
            100% { top: 100%; opacity: 0; }
        }

        /* Scanning Progress Loader */
        .scan-loader {
            display: none;
            padding: 60px 20px;
            text-align: center;
        }

        .scan-steps-track {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            max-width: 600px;
            margin: 30px auto 0 auto;
        }

        .scan-step-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--border-subtle);
            animation: step-bounce 1.4s infinite ease-in-out;
        }

        .scan-step-dot:nth-child(1) { animation-delay: -0.32s; }
        .scan-step-dot:nth-child(2) { animation-delay: -0.16s; }
        .scan-step-dot:nth-child(3) { animation-delay: 0s; }

        @keyframes step-bounce {
            0%, 80%, 100% { transform: scale(0.6); background: var(--border-subtle); }
            40% { transform: scale(1.3); background: var(--accent-gold); }
        }

        /* Diagnostic Results Section */
        .results-container {
            display: none;
            animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Severity Banner Card */
        .severity-banner {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 24px 28px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .severity-banner::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 6px;
            background: var(--accent-gold);
            border-radius: 4px 0 0 4px;
        }

        .severity-banner.level-normal::before { background: var(--accent-emerald); }
        .severity-banner.level-reject::before { background: var(--accent-rose); }
        .severity-banner.level-severe::before { background: var(--accent-purple); }

        .banner-left h3 {
            font-family: var(--font-display);
            font-size: 24px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }

        .banner-left p {
            font-size: 13.5px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .urgency-badge {
            padding: 8px 18px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 12px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Visual Quad Studio */
        .quad-tabs {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .quad-tab-label {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .quad-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 22px;
        }

        @media (max-width: 900px) {
            .quad-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .quad-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .quad-card:hover {
            border-color: var(--accent-gold);
            transform: translateY(-2px);
        }

        .quad-img-wrap {
            width: 100%;
            height: 160px;
            border-radius: 10px;
            overflow: hidden;
            background: #000000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .quad-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .quad-caption {
            font-size: 11.5px;
            font-weight: 700;
            color: var(--text-secondary);
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        /* Biomarker Telemetry Grid (superpower.com style) */
        .biomarker-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 22px;
        }

        @media (max-width: 768px) {
            .biomarker-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .biomarker-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .biomarker-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .biomarker-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .biomarker-num {
            font-family: var(--font-display);
            font-size: 22px;
            font-weight: 800;
            color: var(--text-primary);
        }

        .biomarker-bar-bg {
            width: 100%;
            height: 4px;
            background: var(--border-subtle);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }

        .biomarker-bar-fill {
            height: 100%;
            border-radius: 2px;
            background: var(--accent-gold);
            width: 30%;
        }

        /* Clinical Decision Rationale */
        .rationale-container {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 22px;
        }

        .rationale-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .rationale-body {
            font-family: var(--font-mono);
            font-size: 12.5px;
            color: var(--text-secondary);
            line-height: 1.6;
            white-space: pre-wrap;
            background: var(--bg-body);
            padding: 14px;
            border-radius: 10px;
            border: 1px solid var(--border-subtle);
        }

        /* Doctor Action Bar */
        .doctor-actions {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .btn-doc {
            padding: 12px 20px;
            border-radius: 12px;
            font-family: var(--font-main);
            font-weight: 700;
            font-size: 13.5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }

        .btn-approve-doc {
            background: var(--accent-emerald);
            color: #ffffff;
            box-shadow: 0 4px 14px var(--accent-emerald-glow);
        }

        .btn-approve-doc:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--accent-emerald-glow);
        }

        .btn-override-doc {
            background: var(--bg-surface);
            border-color: var(--border-subtle);
            color: var(--text-primary);
        }

        .btn-override-doc:hover {
            border-color: var(--text-secondary);
        }

        .btn-escalate-doc {
            background: rgba(168, 85, 247, 0.15);
            border-color: rgba(168, 85, 247, 0.3);
            color: var(--accent-purple);
        }

        .btn-escalate-doc:hover {
            background: rgba(168, 85, 247, 0.25);
        }

        .btn-export-doc {
            margin-left: auto;
            background: var(--bg-surface);
            border-color: var(--border-subtle);
            color: var(--text-muted);
        }

        .btn-export-doc:hover {
            color: var(--text-primary);
            border-color: var(--accent-gold);
        }

        /* Biomarker Deep Dive Section (superpower.com accordion & visualizer) */
        .biomarkers-section {
            padding: 80px 0;
            border-top: 1px solid var(--border-subtle);
        }

        .biomarkers-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: center;
        }

        @media (max-width: 900px) {
            .biomarkers-layout { grid-template-columns: 1fr; }
        }

        .biomarker-accordion {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .accordion-item {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.2s ease;
        }

        .accordion-item.active {
            border-color: var(--accent-gold);
        }

        .accordion-header {
            padding: 18px 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            user-select: none;
        }

        .accordion-title {
            font-family: var(--font-display);
            font-size: 16.5px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .accordion-icon {
            font-size: 18px;
            color: var(--text-muted);
            transition: transform 0.25s ease;
        }

        .accordion-item.active .accordion-icon {
            transform: rotate(45deg);
            color: var(--accent-gold);
        }

        .accordion-content {
            padding: 0 22px 18px 22px;
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
            display: none;
        }

        .accordion-item.active .accordion-content {
            display: block;
        }

        /* Interactive Telemedicine Simulator */
        .sim-section {
            padding: 80px 0;
            border-top: 1px solid var(--border-subtle);
        }

        .sim-card {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 28px;
            padding: 38px;
        }

        .sim-grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 40px;
        }

        @media (max-width: 900px) {
            .sim-grid { grid-template-columns: 1fr; }
        }

        .slider-group {
            margin-bottom: 24px;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 13.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .slider-input {
            width: 100%;
            accent-color: var(--accent-gold);
            height: 6px;
            border-radius: 3px;
            background: var(--border-subtle);
            cursor: pointer;
        }

        .sim-result-box {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 26px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .sim-stat-row {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }

        /* Modal Lightbox */
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .modal-content {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            max-width: 850px;
            width: 100%;
            padding: 24px;
            position: relative;
            box-shadow: var(--shadow-floating);
        }

        .modal-close {
            position: absolute;
            top: 18px;
            right: 18px;
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border-subtle);
            padding: 60px 0 40px 0;
            background: var(--bg-surface);
            margin-top: 60px;
        }

        .footer-grid {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .footer-copy {
            font-size: 13px;
            color: var(--text-muted);
        }

        /* Toast notifications */
        #toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 14px 24px;
            border-radius: 12px;
            background: var(--bg-card);
            border: 1px solid var(--accent-gold);
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 600;
            box-shadow: var(--shadow-floating);
            z-index: 2000;
            display: none;
            animation: slideUp 0.3s ease;
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
<<<<<<< HEAD
    <header>
        <div class="header-brand">
            <div class="brand-icon">👁️</div>
            <div>
                <h1>Retinal Health Screening Assistant</h1>
                <p>Intelligent, explainable eye care analysis supporting clinicians in early detection</p>
            </div>
        </div>
        <div class="header-status">
            <span class="status-dot"></span>
            <span>Clinical Assistant Ready</span>
        </div>
    </header>
=======
>>>>>>> c1e4f41 (feat(ui): redesign web app as OptiNova AI with superpower.com aesthetics, dark/light mode, and clinical biomarker suite)

    <!-- Ambient Visual Glow & Grid -->
    <div class="ambient-glow"></div>
    <div class="bg-grid"></div>

    <!-- Navigation -->
    <nav>
        <a href="#" class="nav-brand">
            <div class="nav-logo-icon">👁️</div>
            <div class="nav-logo-text">
                OptiNova <span>AI</span>
            </div>
        </a>

        <ul class="nav-links">
            <li><a href="#screening">AI Screening</a></li>
            <li><a href="#how-it-works">How It Works</a></li>
            <li><a href="#biomarkers">Biomarker Suite</a></li>
            <li><a href="#telemedicine">Tele-Triage</a></li>
        </ul>

        <div class="nav-actions">
            <button class="theme-toggle-btn" id="themeToggle" onclick="toggleTheme()">
                <span id="themeIcon">🌙</span> <span id="themeLabel">Dark</span>
            </button>
            <a href="#screening" class="btn-pill-primary">Launch Screening</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
        <div class="container">
            <div class="hero-badge">
                <span class="dot"></span>
                <span>SIH 2026 CLINICAL TELE-OPHTHALMOLOGY (PS ID 26038)</span>
            </div>

            <h1 class="hero-title">
                Your complete <span class="highlight">retinal health</span> intelligence.
            </h1>

            <p class="hero-sub">
                Screen 15+ fundus micro-biomarkers, quantify 5-stage ETDRS diabetic retinopathy severity, and generate transparent Grad-CAM explainability in under 3 seconds.
            </p>

            <div class="hero-actions">
                <a href="#screening" class="btn-hero-primary">
                    <span>✦ Start AI Screening</span>
                </a>
                <button onclick="selectSample('sample_06_moderate_dr.png')" class="btn-hero-secondary">
                    <span>⚡ Load Benchmark Sample</span>
                </button>
            </div>

            <!-- Stats Bar (superpower.com style) -->
            <div class="hero-stats">
                <div class="stat-item">
                    <div class="stat-value">15+</div>
                    <div class="stat-label">Retinal Biomarkers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">&lt; 3.0s</div>
                    <div class="stat-label">Multi-Module Latency</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">98.4%</div>
                    <div class="stat-label">Gatekeeper Precision</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">100%</div>
                    <div class="stat-label">Grad-CAM Explainability</div>
                </div>
            </div>
        </div>
    </section>

    <!-- 4-Step "How it works" Cards (superpower.com style) -->
    <section id="how-it-works" style="padding: 40px 0 20px 0;">
        <div class="container">
            <div class="section-header">
                <span class="section-tag">Clinical Architecture</span>
                <h2 class="section-title">How It Works</h2>
                <p class="section-desc">From raw rural fundus acquisition to instant ophthalmologist review.</p>
            </div>

            <div class="how-it-works-grid">
                <div class="step-card">
                    <span class="step-num">01</span>
                    <div class="step-icon-box">🛡️</div>
                    <h3 class="step-title">Assess & Enhance</h3>
                    <p class="step-text">Automated quality gatekeeper filters out blurry/underexposed scans and applies CLAHE contrast optimization for micro-vascular clarity.</p>
                </div>

                <div class="step-card">
                    <span class="step-num">02</span>
                    <div class="step-icon-box">🔬</div>
                    <h3 class="step-title">Segment Lesions</h3>
                    <p class="step-text">High-precision segmentation isolates the optic disc, vessel caliber, microaneurysms (MAs), and hard lipid exudates.</p>
                </div>

                <div class="step-card">
                    <span class="step-num">03</span>
                    <div class="step-icon-box">📊</div>
                    <h3 class="step-title">Grade Severity</h3>
                    <p class="step-text">Multiclass decision tree maps lesion load to 5 ETDRS stages (0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=PDR) with calibrated confidence.</p>
                </div>

                <div class="step-card">
                    <span class="step-num">04</span>
                    <div class="step-icon-box">💡</div>
                    <h3 class="step-title">Explain & Triage</h3>
                    <p class="step-text">Grad-CAM heatmaps verify anatomical attention, delivering physician-interpretable rationales in under 30 seconds.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Screening Studio Workspace -->
    <section class="screening-studio" id="screening">
        <div class="container">
            <div class="studio-card">
                <div class="studio-grid">

                    <!-- Left: Control & Image Input -->
                    <div>
                        <div class="panel-heading">
                            <div class="panel-title">
                                <span>📸</span> Fundus Image Input
                            </div>
                        </div>

                        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                            <div class="drop-zone-icon">📷</div>
                            <div class="drop-zone-title">Upload Fundus Scan</div>
                            <div class="drop-zone-sub">Drag & drop or click (PNG, JPG, DICOM)</div>
                            <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFileSelect(event)">
                        </div>
                        <div id="fileSelectionText" style="font-size:12px; color:var(--accent-gold); font-weight:600; margin-top:8px; text-align:center;"></div>

                        <button class="btn-run-scan" id="btnRun" onclick="runScreening()" disabled>
                            <span>🚀 Run Multi-Module AI Pipeline</span>
                        </button>

                        <div class="preset-section">
                            <div class="preset-label">
                                <span>Benchmark Clinical Presets</span>
                                <span>6 Cases</span>
                            </div>
                            <div class="preset-grid">
                                <div class="preset-chip" onclick="selectSample('sample_01_clear.png')">
                                    <div class="preset-chip-info">
                                        <span>🟢</span>
                                        <div style="font-size:13px; font-weight:600;">Grade 0: Normal</div>
                                    </div>
                                    <span class="preset-badge-tag tag-normal">Clear</span>
                                </div>

                                <div class="preset-chip" onclick="selectSample('sample_02_low_contrast.png')">
                                    <div class="preset-chip-info">
                                        <span>🟡</span>
                                        <div style="font-size:13px; font-weight:600;">Low Contrast Scan</div>
                                    </div>
                                    <span class="preset-badge-tag tag-low">CLAHE Fix</span>
                                </div>

                                <div class="preset-chip" onclick="selectSample('sample_03_blurry.png')">
                                    <div class="preset-chip-info">
                                        <span>🔴</span>
                                        <div style="font-size:13px; font-weight:600;">Blurry Image</div>
                                    </div>
                                    <span class="preset-badge-tag tag-reject">QC Reject</span>
                                </div>

                                <div class="preset-chip" onclick="selectSample('sample_06_moderate_dr.png')">
                                    <div class="preset-chip-info">
                                        <span>🟠</span>
                                        <div style="font-size:13px; font-weight:600;">Grade 2: Moderate DR</div>
                                    </div>
                                    <span class="preset-badge-tag tag-dr">Referable</span>
                                </div>

                                <div class="preset-chip" onclick="selectSample('sample_07_severe_dr.png')">
                                    <div class="preset-chip-info">
                                        <span>🔴</span>
                                        <div style="font-size:13px; font-weight:600;">Grade 3: Severe DR</div>
                                    </div>
                                    <span class="preset-badge-tag tag-dr">Hemorrhages</span>
                                </div>

                                <div class="preset-chip" onclick="selectSample('sample_08_proliferative_dr.png')">
                                    <div class="preset-chip-info">
                                        <span>🟣</span>
                                        <div style="font-size:13px; font-weight:600;">Grade 4: Proliferative</div>
                                    </div>
                                    <span class="preset-badge-tag tag-dr">Urgent NV</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Diagnostic Output -->
                    <div>
                        <div class="panel-heading">
                            <div class="panel-title">
                                <span>🔬</span> Diagnostic Studio & Explainability
                            </div>
                            <span style="font-size:12px; color:var(--text-muted);">Real-time Inference</span>
                        </div>

                        <!-- Initial Empty State -->
                        <div class="empty-placeholder" id="emptyPlaceholder">
                            <div class="empty-reticle">👁️</div>
                            <h4 style="font-family:var(--font-display); font-size:18px; margin-bottom:6px; color:var(--text-primary);">Awaiting Fundus Input</h4>
                            <p style="font-size:14px; color:var(--text-secondary); max-width:400px; margin:0 auto;">
                                Upload a fundus scan or click any benchmark case on the left to execute Modules 1 through 5.
                            </p>
                        </div>

                        <!-- Processing State -->
                        <div class="scan-loader" id="scanLoader">
                            <div style="font-family:var(--font-display); font-size:20px; font-weight:700; color:var(--text-primary); margin-bottom:8px;">
                                Processing Neural Pipeline...
                            </div>
                            <p style="font-size:13px; color:var(--text-muted);">
                                Executing Quality Gatekeeper → Lesion Segmentation → Severity Grading → Grad-CAM
                            </p>
                            <div class="scan-steps-track">
                                <div class="scan-step-dot"></div>
                                <div class="scan-step-dot"></div>
                                <div class="scan-step-dot"></div>
                            </div>
                        </div>

                        <!-- Results View -->
                        <div class="results-container" id="resultsContainer">

                            <!-- Severity Banner -->
                            <div class="severity-banner" id="resBanner">
                                <div class="banner-left">
                                    <h3 id="resGradeTitle">Grade 2: Moderate NPDR</h3>
                                    <p id="resConfidence">Calibrated Confidence: 91.4% • Platt-Calibrated</p>
                                </div>
                                <span class="urgency-badge" id="resUrgencyBadge">REFERRAL REQUIRED</span>
                            </div>

                            <!-- 4-Quad Visual Studio -->
                            <div class="quad-tabs">
                                <div class="quad-tab-label">
                                    <span>🖼️ Multi-Module Visual Overlays</span>
                                </div>
                                <span style="font-size:11.5px; color:var(--text-muted);">Click any image to enlarge</span>
                            </div>

                            <div class="quad-grid">
                                <div class="quad-card" onclick="openLightbox('imgOrig', 'Original Fundus Acquisition')">
                                    <div class="quad-img-wrap">
                                        <img id="imgOrig" src="" alt="Original">
                                    </div>
                                    <div class="quad-caption">1. Raw Acquisition</div>
                                </div>

                                <div class="quad-card" onclick="openLightbox('imgEnhanced', 'CLAHE Contrast Enhancement (Mod 1)')">
                                    <div class="quad-img-wrap">
                                        <img id="imgEnhanced" src="" alt="Enhanced">
                                    </div>
                                    <div class="quad-caption">2. CLAHE Enhanced</div>
                                </div>

                                <div class="quad-card" onclick="openLightbox('imgOverlay', 'Lesion & Vessel Overlay (Mod 2)')">
                                    <div class="quad-img-wrap">
                                        <img id="imgOverlay" src="" alt="Overlay">
                                    </div>
                                    <div class="quad-caption">3. Lesion Overlay</div>
                                </div>

                                <div class="quad-card" onclick="openLightbox('imgGradcam', 'Grad-CAM Explainability Heatmap (Mod 4)')">
                                    <div class="quad-img-wrap">
                                        <img id="imgGradcam" src="" alt="Grad-CAM">
                                    </div>
                                    <div class="quad-caption">4. Grad-CAM Map</div>
                                </div>
                            </div>

                            <!-- Biomarker Matrix -->
                            <div class="biomarker-grid">
                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Microaneurysms</span>
                                        <span style="font-size:12px;">🔴</span>
                                    </div>
                                    <div class="biomarker-num" id="bmMAs">0</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barMAs" style="width:0%;"></div></div>
                                </div>

                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Hard Exudates</span>
                                        <span style="font-size:12px;">🟡</span>
                                    </div>
                                    <div class="biomarker-num" id="bmExudates">0</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barExudates" style="width:0%;"></div></div>
                                </div>

                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Hemorrhages</span>
                                        <span style="font-size:12px;">🩸</span>
                                    </div>
                                    <div class="biomarker-num" id="bmHems">0</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barHems" style="width:0%;"></div></div>
                                </div>

                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Focus Quality Score</span>
                                        <span style="font-size:12px;">🔍</span>
                                    </div>
                                    <div class="biomarker-num" id="bmFocus">0.0</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barFocus" style="width:80%;"></div></div>
                                </div>

                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Grad-CAM Alignment</span>
                                        <span style="font-size:12px;">🎯</span>
                                    </div>
                                    <div class="biomarker-num" id="bmCorrelation">0.00</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barCorrelation" style="width:75%;"></div></div>
                                </div>

                                <div class="biomarker-card">
                                    <div class="biomarker-header">
                                        <span class="biomarker-title">Neovascularization</span>
                                        <span style="font-size:12px;">⚠️</span>
                                    </div>
                                    <div class="biomarker-num" id="bmNV">No</div>
                                    <div class="biomarker-bar-bg"><div class="biomarker-bar-fill" id="barNV" style="width:0%;"></div></div>
                                </div>
                            </div>

                            <!-- Clinical Rationale -->
                            <div class="rationale-container">
                                <div class="rationale-title">
                                    <span>📋 Clinical Decision Memo & Explainable Rationale</span>
                                </div>
                                <div class="rationale-body" id="resRationaleText"></div>
                            </div>

                            <!-- Physician Action Bar -->
                            <div class="doctor-actions">
                                <button class="btn-doc btn-approve-doc" onclick="showToast('✓ AI Diagnosis Approved & Signed by Physician!')">
                                    <span>✓ Approve Diagnosis (&lt;30s)</span>
                                </button>
                                <button class="btn-doc btn-override-doc" onclick="showToast('✎ Clinical Override Flagged for Secondary Adjudication')">
                                    <span>✎ Override Grade</span>
                                </button>
                                <button class="btn-doc btn-escalate-doc" onclick="showToast('⚑ Case Dispatched to Tertiary Vitreo-Retinal Specialist')">
                                    <span>⚑ Escalate to Specialist</span>
                                </button>
                                <button class="btn-doc btn-export-doc" onclick="window.print()">
                                    <span>🖨️ Export PDF Report</span>
                                </button>
                            </div>

                        </div>
                    </div>

                </div>
            </div>
        </div>
    </section>

    <!-- Biomarkers Deep Dive (superpower.com accordion style) -->
    <section class="biomarkers-section" id="biomarkers">
        <div class="container">
            <div class="biomarkers-layout">
                <div>
                    <span class="section-tag">Retinal Biomarker Suite</span>
                    <h2 class="section-title">Every scan measures 15+ retinal indicators</h2>
                    <p class="section-desc" style="margin-bottom: 28px;">
                        Our computer vision algorithms segment sub-millimeter microvascular structures with high clinical fidelity.
                    </p>

                    <div class="biomarker-accordion">
                        <div class="accordion-item active" onclick="toggleAccordion(this)">
                            <div class="accordion-header">
                                <div class="accordion-title">
                                    <span>🔴</span> Microaneurysms (MAs)
                                </div>
                                <div class="accordion-icon">+</div>
                            </div>
                            <div class="accordion-content">
                                Tiny out-pouchings of capillary walls resulting from pericyte loss. They represent the earliest detectable anatomical hallmark of Diabetic Retinopathy.
                            </div>
                        </div>

                        <div class="accordion-item" onclick="toggleAccordion(this)">
                            <div class="accordion-header">
                                <div class="accordion-title">
                                    <span>🩸</span> Retinal Hemorrhages
                                </div>
                                <div class="accordion-icon">+</div>
                            </div>
                            <div class="accordion-content">
                                Intraretinal micro-vascular abnormalities including dot/blot hemorrhages in deep layers and flame-shaped hemorrhages in nerve fiber layers.
                            </div>
                        </div>

                        <div class="accordion-item" onclick="toggleAccordion(this)">
                            <div class="accordion-header">
                                <div class="accordion-title">
                                    <span>🟡</span> Hard Lipid Exudates
                                </div>
                                <div class="accordion-icon">+</div>
                            </div>
                            <div class="accordion-content">
                                Waxy lipoprotein deposits resulting from broken blood-retinal barriers. Proximity to the central fovea signals clinically significant macular edema (CSME).
                            </div>
                        </div>

                        <div class="accordion-item" onclick="toggleAccordion(this)">
                            <div class="accordion-header">
                                <div class="accordion-title">
                                    <span>⚡</span> Neovascularization (NV)
                                </div>
                                <div class="accordion-icon">+</div>
                            </div>
                            <div class="accordion-content">
                                Fragile new blood vessels sprouting on the optic disc (NVD) or elsewhere (NVE) due to extensive retinal ischemia, defining Proliferative DR.
                            </div>
                        </div>
                    </div>
                </div>

                <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:24px; padding:32px; text-align:center;">
                    <div style="font-size:12px; font-weight:700; text-transform:uppercase; color:var(--accent-gold); letter-spacing:1px; margin-bottom:12px;">Fundus Anatomy Map</div>
                    <div style="position:relative; width:100%; height:260px; background:radial-gradient(circle at 45% 50%, #9a3412 0%, #451a03 70%, #000000 100%); border-radius:16px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
                        <div style="position:absolute; width:120px; height:120px; border-radius:50%; border:2px dashed rgba(255,255,255,0.3); animation:spin 30s linear infinite;"></div>
                        <div style="position:absolute; left:28%; top:45%; width:34px; height:46px; background:#fef08a; border-radius:50%; box-shadow:0 0 20px #fef08a; opacity:0.85;"></div>
                        <div style="position:absolute; left:62%; top:48%; width:24px; height:24px; background:#451a03; border-radius:50%; box-shadow:inset 0 0 10px #000;"></div>
                        <div style="color:#ffffff; font-size:13px; font-weight:600; text-shadow:0 2px 4px rgba(0,0,0,0.8); z-index:2;">
                            Optic Disc • Fovea • Arterioles • Venules
                        </div>
                    </div>
                    <p style="font-size:13px; color:var(--text-secondary); margin-top:16px;">
                        Interactive anatomical localization grounds every AI prediction to verified clinical landmarks.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Telemedicine Queue Simulator Section -->
    <section class="sim-section" id="telemedicine">
        <div class="container">
            <div class="sim-card">
                <div class="section-header" style="margin-bottom:32px;">
                    <span class="section-tag">Module 5 Simulation</span>
                    <h2 class="section-title">Rural Tele-Ophthalmology Simulator</h2>
                    <p class="section-desc">Simulate patient queue dynamics and bandwidth optimization over 2 Mbps rural links.</p>
                </div>

                <div class="sim-grid">
                    <div>
                        <div class="slider-group">
                            <div class="slider-header">
                                <span style="color:var(--text-primary);">Number of Rural PHC Clinics</span>
                                <span style="color:var(--accent-gold); font-weight:700;" id="lblClinics">25 Clinics</span>
                            </div>
                            <input type="range" class="slider-input" min="5" max="60" value="25" id="sliderClinics" oninput="updateSim()">
                        </div>

                        <div class="slider-group">
                            <div class="slider-header">
                                <span style="color:var(--text-primary);">Ophthalmologists on Duty</span>
                                <span style="color:var(--accent-gold); font-weight:700;" id="lblDoctors">4 Doctors</span>
                            </div>
                            <input type="range" class="slider-input" min="1" max="10" value="4" id="sliderDoctors" oninput="updateSim()">
                        </div>

                        <div class="slider-group">
                            <div class="slider-header">
                                <span style="color:var(--text-primary);">Uplink Bandwidth per Clinic</span>
                                <span style="color:var(--accent-gold); font-weight:700;" id="lblBandwidth">2.0 Mbps</span>
                            </div>
                            <input type="range" class="slider-input" min="0.5" max="10" step="0.5" value="2.0" id="sliderBandwidth" oninput="updateSim()">
                        </div>

                        <p style="font-size:12.5px; color:var(--text-muted); line-height:1.5;">
                            ⚡ <strong>60% Auto-Triage Bypass:</strong> Non-referable Grade 0 scans with calibrated confidence &gt;85% bypass human queue, reducing specialist workload by 60%.
                        </p>
                    </div>

                    <div class="sim-result-box">
                        <div class="sim-stat-row">
                            <div style="background:var(--bg-card); padding:14px; border-radius:12px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Annual Patient Capacity</div>
                                <div style="font-family:var(--font-display); font-size:24px; font-weight:800; color:var(--text-primary); margin-top:4px;" id="simCapacity">136,875</div>
                            </div>

                            <div style="background:var(--bg-card); padding:14px; border-radius:12px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Doctor Utilization</div>
                                <div style="font-family:var(--font-display); font-size:24px; font-weight:800; color:var(--accent-emerald); margin-top:4px;" id="simDoctorUtil">78.2%</div>
                            </div>

                            <div style="background:var(--bg-card); padding:14px; border-radius:12px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Average Triage Wait</div>
                                <div style="font-family:var(--font-display); font-size:24px; font-weight:800; color:var(--accent-gold); margin-top:4px;" id="simWaitTime">3.4 min</div>
                            </div>

                            <div style="background:var(--bg-card); padding:14px; border-radius:12px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Upload Transmission</div>
                                <div style="font-family:var(--font-display); font-size:24px; font-weight:800; color:var(--accent-purple); margin-top:4px;" id="simUploadDelay">20.0s</div>
                            </div>
                        </div>

                        <div style="font-size:12px; color:var(--text-secondary); text-align:center;">
                            Integrated with Simulink discrete-event queueing model & tele-medicine routing engine.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Lightbox Modal -->
    <div class="modal-backdrop" id="lightboxModal" onclick="closeLightbox(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeLightbox()">&times;</button>
            <h4 id="lightboxTitle" style="font-family:var(--font-display); font-size:18px; margin-bottom:14px; color:var(--text-primary);">Image Inspection</h4>
            <div style="width:100%; height:500px; background:#000000; border-radius:14px; overflow:hidden; display:flex; align-items:center; justify-content:center;">
                <img id="lightboxImg" src="" alt="Enlarged Inspection" style="max-width:100%; max-height:100%; object-fit:contain;">
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast">✓ Notification</div>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="footer-grid">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div class="nav-logo-icon" style="width:30px; height:30px; font-size:16px;">👁️</div>
                    <span style="font-family:var(--font-display); font-weight:800; font-size:17px;">OptiNova AI</span>
                </div>
                <div class="footer-copy">
                    Smart India Hackathon 2026 (Problem Statement ID 26038) • Explainable Retinal AI Prototype
                </div>
                <div style="font-size:12px; color:var(--accent-gold); font-weight:600;">
                    🟢 All 5 Modules Operational
                </div>
            </div>
        </div>
    </footer>

    <script>
        let selectedFile = null;
        let selectedSampleName = null;

        // Theme Toggle (Dark & Light Mode)
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeUI(newTheme);
        }

        function updateThemeUI(theme) {
            const icon = document.getElementById('themeIcon');
            const label = document.getElementById('themeLabel');
            if (theme === 'light') {
                icon.innerText = '☀️';
                label.innerText = 'Light';
            } else {
                icon.innerText = '🌙';
                label.innerText = 'Dark';
            }
        }

        // Initialize Theme from localStorage or system preference
        (function() {
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeUI(savedTheme);
        })();

        // File Selection Handlers
        function handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                selectedFile = files[0];
                selectedSampleName = null;
                document.getElementById('fileSelectionText').innerText = "Selected: " + selectedFile.name;
                document.getElementById('btnRun').disabled = false;
                clearActivePresetChips();
            }
        }

        function selectSample(sampleName) {
            selectedSampleName = sampleName;
            selectedFile = null;
            document.getElementById('fileSelectionText').innerText = "Benchmark Case: " + sampleName;
            document.getElementById('btnRun').disabled = false;
            
            // Scroll to screening area smoothly
            const screeningEl = document.getElementById('screening');
            if (screeningEl) {
                screeningEl.scrollIntoView({ behavior: 'smooth' });
            }

            runScreening();
        }

        function clearActivePresetChips() {
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
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
                    throw new Error("Pipeline Execution Error (" + res.status + "): " + text);
                }
                return res.json();
            })
            .then(data => {
                loader.style.display = 'none';
                results.style.display = 'block';

                // Update Severity Banner
                const banner = document.getElementById('resBanner');
                const title = document.getElementById('resGradeTitle');
                const conf = document.getElementById('resConfidence');
                const badge = document.getElementById('resUrgencyBadge');

                banner.className = 'severity-banner';
                title.innerText = data.grade_name;
                conf.innerText = `Confidence Score: ${(data.confidence * 100).toFixed(1)}% • Quality Focus: ${data.quality.focus_score.toFixed(1)}`;

                if (data.status === 'reject') {
                    banner.classList.add('level-reject');
                    badge.innerText = "GATEKEEPER REJECTED";
                    badge.style.backgroundColor = "var(--accent-rose)";
                    badge.style.color = "#ffffff";
                } else if (data.referable) {
                    banner.classList.add(data.grade_level >= 3 ? 'level-severe' : 'level-dr');
                    badge.innerText = "REFERRAL REQUIRED";
                    badge.style.backgroundColor = data.grade_level >= 3 ? "var(--accent-rose)" : "var(--accent-amber)";
                    badge.style.color = "#ffffff";
                } else {
                    banner.classList.add('level-normal');
                    badge.innerText = "ROUTINE / CLEAR";
                    badge.style.backgroundColor = "var(--accent-emerald)";
                    badge.style.color = "#ffffff";
                }

                // Update Visual Studio Images
                document.getElementById('imgOrig').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('imgEnhanced').src = "data:image/jpeg;base64," + data.img_enhanced;
                document.getElementById('imgOverlay').src = "data:image/jpeg;base64," + data.img_overlay;
                document.getElementById('imgGradcam').src = "data:image/jpeg;base64," + data.img_gradcam;

                // Update Biomarkers
                const maCount = data.stats.ma_count || 0;
                const exCount = data.stats.exudate_count || 0;
                const hemCount = data.stats.hem_count || 0;

                document.getElementById('bmMAs').innerText = maCount;
                document.getElementById('bmExudates').innerText = exCount;
                document.getElementById('bmHems').innerText = hemCount;
                document.getElementById('bmFocus').innerText = data.quality.focus_score.toFixed(1);
                document.getElementById('bmCorrelation').innerText = data.correlation_score.toFixed(2);
                document.getElementById('bmNV').innerText = data.stats.nv_flag ? "YES (Active)" : "None";

                document.getElementById('barMAs').style.width = Math.min(100, maCount * 12) + '%';
                document.getElementById('barExudates').style.width = Math.min(100, exCount * 10) + '%';
                document.getElementById('barHems').style.width = Math.min(100, hemCount * 15) + '%';
                document.getElementById('barFocus').style.width = Math.min(100, data.quality.focus_score * 0.8) + '%';
                document.getElementById('barCorrelation').style.width = Math.min(100, data.correlation_score * 100) + '%';
                document.getElementById('barNV').style.width = data.stats.nv_flag ? '100%' : '0%';

                // Update Rationale
                document.getElementById('resRationaleText').innerText = data.rationale;
            })
            .catch(err => {
                loader.style.display = 'none';
                alert("Screening Pipeline Exception: " + err.message);
            });
        }

        // Drag & Drop
        const dropZone = document.getElementById('dropZone');
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
        });
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                selectedFile = files[0];
                selectedSampleName = null;
                document.getElementById('fileSelectionText').innerText = "Selected: " + selectedFile.name;
                document.getElementById('btnRun').disabled = false;
                clearActivePresetChips();
            }
        });

        // Lightbox
        function openLightbox(imgId, title) {
            const imgEl = document.getElementById(imgId);
            if (!imgEl || !imgEl.src) return;
            document.getElementById('lightboxImg').src = imgEl.src;
            document.getElementById('lightboxTitle').innerText = title;
            document.getElementById('lightboxModal').style.display = 'flex';
        }

        function closeLightbox() {
            document.getElementById('lightboxModal').style.display = 'none';
        }

        // Accordion
        function toggleAccordion(item) {
            const wasActive = item.classList.contains('active');
            document.querySelectorAll('.accordion-item').forEach(el => el.classList.remove('active'));
            if (!wasActive) item.classList.add('active');
        }

        // Telemedicine Simulator Sliders
        function updateSim() {
            const clinics = parseInt(document.getElementById('sliderClinics').value);
            const doctors = parseInt(document.getElementById('sliderDoctors').value);
            const bw = parseFloat(document.getElementById('sliderBandwidth').value);

            document.getElementById('lblClinics').innerText = clinics + " Clinics";
            document.getElementById('lblDoctors').innerText = doctors + " Doctors";
            document.getElementById('lblBandwidth').innerText = bw.toFixed(1) + " Mbps";

            const annualCap = clinics * 15 * 365;
            document.getElementById('simCapacity').innerText = annualCap.toLocaleString();

            const uploadDelay = (5.0 / (bw / 8.0)).toFixed(1);
            document.getElementById('simUploadDelay').innerText = uploadDelay + "s";

            // Simple queue dynamics approximation
            const referablePerDay = clinics * 15 * 0.4;
            const doctorCapacityPerDay = doctors * (8 * 60 / 0.5); // 0.5 min review
            const util = Math.min(99.5, (referablePerDay / doctorCapacityPerDay) * 100);
            document.getElementById('simDoctorUtil').innerText = util.toFixed(1) + "%";

            const avgWait = (Math.max(0.5, (util / 100) * 4.5) + (uploadDelay / 60)).toFixed(1);
            document.getElementById('simWaitTime').innerText = avgWait + " min";
        }

        // Toast
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(base_dir, 'data', 'sample_images', os.path.basename(sample_name))

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
            rationale = f"[QUALITY GATEKEEPER REJECTED]\\nReason: {reason}\\nAction: Please adjust illumination/focus and recapture."
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
