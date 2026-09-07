#!/usr/bin/env python3
"""
OptiNova AI — Explainable Retinal Intelligence & Diabetic Retinopathy Screening (SIH 2026)
Smart India Hackathon 2026 | Problem Statement ID: SIH26038 | Theme: MedTech / Clean & Green Software
Team: Optinova | Hardware: Zero-CAPEX Edge (x86 / ARM / Raspberry Pi 4 / INT8 Quantized)
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

def image_to_base64(img_bgr, quality=85):
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
    <title>OptiNova AI | Explainable Retinal Intelligence • SIH 2026</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-display: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;

            /* Dark Theme Variables (Default) */
            --bg-body: #07080c;
            --bg-surface: #0e111a;
            --bg-surface-elevated: #141824;
            --bg-card: rgba(16, 20, 31, 0.75);
            --bg-card-hover: rgba(23, 29, 45, 0.9);
            --bg-badge: rgba(255, 255, 255, 0.06);
            --border-subtle: rgba(255, 255, 255, 0.09);
            --border-active: rgba(245, 158, 11, 0.5);
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;

            --accent-gold: #f59e0b;
            --accent-gold-glow: rgba(245, 158, 11, 0.3);
            --accent-amber: #d97706;
            --accent-emerald: #10b981;
            --accent-emerald-glow: rgba(16, 185, 129, 0.25);
            --accent-rose: #f43f5e;
            --accent-rose-glow: rgba(244, 63, 94, 0.25);
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;

            --hero-glow: radial-gradient(circle at 50% 25%, rgba(245, 158, 11, 0.22) 0%, rgba(234, 88, 12, 0.1) 35%, rgba(7, 8, 12, 0) 70%);
            --hero-glow-secondary: radial-gradient(circle at 80% 60%, rgba(16, 185, 129, 0.12) 0%, transparent 50%);
            --card-glass-blur: blur(24px);
            --shadow-subtle: 0 4px 24px rgba(0, 0, 0, 0.4);
            --shadow-3d: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.07);
            --shadow-card-hover: 0 24px 60px rgba(245, 158, 11, 0.12), 0 0 0 1px rgba(245, 158, 11, 0.4);
        }

        [data-theme="light"] {
            --bg-body: #f8fafc;
            --bg-surface: #ffffff;
            --bg-surface-elevated: #f1f5f9;
            --bg-card: rgba(255, 255, 255, 0.88);
            --bg-card-hover: #ffffff;
            --bg-badge: rgba(15, 23, 42, 0.05);
            --border-subtle: rgba(15, 23, 42, 0.09);
            --border-active: rgba(217, 119, 6, 0.5);

            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;

            --accent-gold: #d97706;
            --accent-gold-glow: rgba(217, 119, 6, 0.2);
            --accent-amber: #b45309;
            --accent-emerald: #059669;
            --accent-emerald-glow: rgba(5, 150, 105, 0.2);
            --accent-rose: #e11d48;
            --accent-rose-glow: rgba(225, 29, 72, 0.2);
            --accent-cyan: #0891b2;
            --accent-purple: #9333ea;

            --hero-glow: radial-gradient(circle at 50% 20%, rgba(251, 191, 36, 0.3) 0%, rgba(253, 230, 138, 0.15) 40%, rgba(248, 250, 252, 0) 70%);
            --hero-glow-secondary: radial-gradient(circle at 80% 50%, rgba(16, 185, 129, 0.12) 0%, transparent 50%);
            --card-glass-blur: blur(24px);
            --shadow-subtle: 0 4px 24px rgba(15, 23, 42, 0.06);
            --shadow-3d: 0 20px 50px rgba(15, 23, 42, 0.09), 0 0 0 1px rgba(15, 23, 42, 0.08);
            --shadow-card-hover: 0 24px 60px rgba(217, 119, 6, 0.12), 0 0 0 1px rgba(217, 119, 6, 0.4);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-body);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* 3D Visual Depth & Perspective Container */
        .perspective-stage {
            perspective: 1200px;
        }

        /* Ambient Lighting Auras */
        .ambient-glow {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 1000px;
            background: var(--hero-glow);
            pointer-events: none;
            z-index: 0;
        }

        .ambient-glow-secondary {
            position: absolute;
            top: 600px;
            right: 0;
            width: 600px;
            height: 800px;
            background: var(--hero-glow-secondary);
            pointer-events: none;
            z-index: 0;
        }

        .bg-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 1100px;
            background-size: 44px 44px;
            background-image: 
                linear-gradient(to right, var(--border-subtle) 1px, transparent 1px),
                linear-gradient(to bottom, var(--border-subtle) 1px, transparent 1px);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, transparent 85%);
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, transparent 85%);
            pointer-events: none;
            z-index: 0;
        }

        /* Top Navigation */
        nav {
            position: sticky;
            top: 0;
            z-index: 200;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 44px;
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

        .nav-logo-3d {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 22px;
            box-shadow: 0 8px 20px var(--accent-gold-glow), inset 0 1px 1px rgba(255,255,255,0.4);
            transform: rotate(-3deg);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .nav-brand:hover .nav-logo-3d {
            transform: rotate(6deg) scale(1.08);
        }

        .nav-logo-text {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 22px;
            letter-spacing: -0.6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-logo-text span {
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            padding: 3px 9px;
            border-radius: 8px;
            background: var(--bg-badge);
            color: var(--accent-gold);
            border: 1px solid var(--border-subtle);
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 26px;
            list-style: none;
        }

        .nav-links a {
            text-decoration: none;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            position: relative;
            padding: 4px 0;
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

        .theme-toggle-btn {
            background: var(--bg-badge);
            border: 1px solid var(--border-subtle);
            border-radius: 30px;
            padding: 7px 16px;
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
            box-shadow: 0 0 15px var(--accent-gold-glow);
        }

        .btn-pitch-deck {
            background: var(--bg-badge);
            color: var(--accent-gold);
            border: 1px solid var(--accent-gold);
            border-radius: 30px;
            padding: 9px 18px;
            font-weight: 700;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: var(--font-main);
        }

        .btn-pitch-deck:hover {
            background: var(--accent-gold);
            color: #ffffff;
            box-shadow: 0 4px 16px var(--accent-gold-glow);
        }

        .btn-pill-primary {
            background: var(--text-primary);
            color: var(--bg-body);
            border: none;
            border-radius: 30px;
            padding: 9px 22px;
            font-weight: 700;
            font-size: 13.5px;
            font-family: var(--font-main);
            cursor: pointer;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
        }

        .btn-pill-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        /* Container */
        .container {
            max-width: 1320px;
            margin: 0 auto;
            padding: 0 24px;
            position: relative;
            z-index: 1;
        }

        /* Hero Section (superpower.com style with 3D elements) */
        .hero {
            padding: 85px 0 50px 0;
            text-align: center;
            position: relative;
        }

        .hero-badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 22px;
            border-radius: 40px;
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-gold);
            margin-bottom: 24px;
            box-shadow: 0 4px 20px var(--accent-gold-glow);
            letter-spacing: 0.4px;
        }

        .hero-badge-pill .pulse-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-gold);
            box-shadow: 0 0 12px var(--accent-gold);
            animation: pulse-ring 2s infinite ease-in-out;
        }

        @keyframes pulse-ring {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.3; transform: scale(0.7); }
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: clamp(40px, 5.8vw, 74px);
            font-weight: 900;
            line-height: 1.06;
            letter-spacing: -2px;
            max-width: 980px;
            margin: 0 auto 22px auto;
        }

        .hero-title .highlight-gold {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 40%, #ea580c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            filter: drop-shadow(0 4px 20px rgba(245, 158, 11, 0.25));
        }

        .hero-sub {
            font-size: 19px;
            color: var(--text-secondary);
            max-width: 740px;
            margin: 0 auto 38px auto;
            font-weight: 400;
            line-height: 1.55;
        }

        .hero-actions {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            margin-bottom: 56px;
            flex-wrap: wrap;
        }

        .btn-hero-3d {
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            color: #ffffff;
            border: none;
            border-radius: 36px;
            padding: 16px 36px;
            font-size: 16px;
            font-weight: 800;
            font-family: var(--font-main);
            cursor: pointer;
            box-shadow: 0 10px 30px var(--accent-gold-glow), inset 0 1px 1px rgba(255,255,255,0.4);
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
        }

        .btn-hero-3d:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 16px 40px var(--accent-gold-glow);
        }

        .btn-hero-glass {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 36px;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            font-family: var(--font-main);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            transition: all 0.25s ease;
        }

        .btn-hero-glass:hover {
            background: var(--bg-card-hover);
            border-color: var(--accent-gold);
            transform: translateY(-2px);
        }

        /* Hero 3D Metrics Strip */
        .hero-stats-card {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            max-width: 1100px;
            margin: 0 auto;
            padding: 26px 30px;
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            box-shadow: var(--shadow-3d);
            transform-style: preserve-3d;
        }

        .stat-column {
            text-align: center;
            border-right: 1px solid var(--border-subtle);
            padding: 6px 14px;
        }

        .stat-column:last-child {
            border-right: none;
        }

        .stat-num {
            font-family: var(--font-display);
            font-size: 32px;
            font-weight: 900;
            color: var(--text-primary);
            letter-spacing: -0.8px;
            line-height: 1.1;
        }

        .stat-subtext {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 6px;
        }

        /* Section Titles */
        .section-header {
            text-align: center;
            margin-bottom: 48px;
        }

        .section-pill-tag {
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--accent-gold);
            margin-bottom: 12px;
            display: inline-block;
            background: var(--bg-badge);
            padding: 4px 14px;
            border-radius: 20px;
            border: 1px solid var(--border-subtle);
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
            max-width: 680px;
            margin: 0 auto;
        }

        /* 3D Retinal Depth Layers Visualization */
        .retina-3d-section {
            padding: 60px 0;
        }

        .depth-card {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 28px;
            padding: 38px;
            box-shadow: var(--shadow-3d);
        }

        .depth-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 40px;
            align-items: center;
        }

        @media (max-width: 980px) {
            .depth-grid { grid-template-columns: 1fr; }
            .hero-stats-card { grid-template-columns: repeat(2, 1fr); }
        }

        .retina-spatial-view {
            position: relative;
            width: 100%;
            height: 380px;
            background: radial-gradient(circle at 50% 50%, #1f0c04 0%, #0c0502 60%, #000000 100%);
            border-radius: 20px;
            border: 1px solid var(--border-subtle);
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            perspective: 800px;
        }

        .spatial-layer {
            position: absolute;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            pointer-events: none;
        }

        .layer-base {
            background: radial-gradient(circle at 45% 50%, rgba(220, 38, 38, 0.8) 0%, rgba(153, 27, 27, 0.9) 60%, rgba(69, 10, 10, 0.95) 100%);
            transform: translateZ(0px) rotateX(15deg);
            box-shadow: 0 0 40px rgba(220, 38, 38, 0.4);
        }

        .layer-vessels {
            border: 2px dashed rgba(254, 240, 138, 0.7);
            transform: translateZ(40px) rotateX(15deg);
            background: radial-gradient(circle at 35% 50%, rgba(254, 240, 138, 0.4) 0%, transparent 40%);
        }

        .layer-lesions {
            border: 2px solid rgba(244, 63, 94, 0.8);
            transform: translateZ(80px) rotateX(15deg);
            box-shadow: 0 0 25px rgba(244, 63, 94, 0.6);
        }

        .layer-gradcam {
            background: radial-gradient(circle at 65% 55%, rgba(168, 85, 247, 0.5) 0%, rgba(6, 182, 212, 0.3) 50%, transparent 80%);
            transform: translateZ(120px) rotateX(15deg);
            mix-blend-mode: screen;
        }

        .depth-controls {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .depth-pill {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .depth-pill:hover, .depth-pill.active {
            border-color: var(--accent-gold);
            background: var(--bg-surface-elevated);
            transform: translateX(4px);
        }

        /* 4-Step "How it works" Cards */
        .how-it-works-section {
            padding: 60px 0 40px 0;
        }

        .how-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 22px;
        }

        @media (max-width: 1024px) {
            .how-grid-4 { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 640px) {
            .how-grid-4 { grid-template-columns: 1fr; }
        }

        .how-3d-card {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 28px 22px;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease, border-color 0.3s ease;
            box-shadow: var(--shadow-subtle);
        }

        .how-3d-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--border-active);
            box-shadow: var(--shadow-card-hover);
        }

        .how-card-num {
            position: absolute;
            top: 20px;
            right: 22px;
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 900;
            color: var(--border-subtle);
        }

        .how-icon-box-3d {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 20px;
            box-shadow: inset 0 1px 1px rgba(255,255,255,0.15);
        }

        .how-card-title {
            font-family: var(--font-display);
            font-size: 19px;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 10px;
        }

        .how-card-text {
            font-size: 13.5px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Main Screening Studio Workspace */
        .screening-studio {
            padding: 50px 0 80px 0;
        }

        .studio-card-3d {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border: 1px solid var(--border-subtle);
            border-radius: 30px;
            padding: 38px;
            box-shadow: var(--shadow-3d);
        }

        .studio-layout {
            display: grid;
            grid-template-columns: 370px 1fr;
            gap: 36px;
        }

        @media (max-width: 1024px) {
            .studio-layout { grid-template-columns: 1fr; }
        }

        .panel-header-sub {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-heading-title {
            font-family: var(--font-display);
            font-size: 18px;
            font-weight: 800;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* 3D Drop Zone */
        .drop-zone-3d {
            border: 2px dashed var(--border-subtle);
            border-radius: 20px;
            padding: 34px 20px;
            text-align: center;
            background: var(--bg-surface);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .drop-zone-3d:hover, .drop-zone-3d.dragover {
            border-color: var(--accent-gold);
            background: var(--bg-badge);
            box-shadow: 0 0 30px var(--accent-gold-glow);
        }

        .drop-zone-icon-3d {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin: 0 auto 14px auto;
            color: var(--accent-gold);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }

        .btn-run-pipeline-3d {
            width: 100%;
            margin-top: 18px;
            padding: 15px;
            background: linear-gradient(135deg, var(--accent-gold), #ea580c);
            color: #ffffff;
            border: none;
            border-radius: 18px;
            font-family: var(--font-main);
            font-weight: 800;
            font-size: 15px;
            cursor: pointer;
            box-shadow: 0 8px 24px var(--accent-gold-glow), inset 0 1px 1px rgba(255,255,255,0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.25s ease;
        }

        .btn-run-pipeline-3d:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px var(--accent-gold-glow);
        }

        .btn-run-pipeline-3d:disabled {
            background: var(--bg-badge);
            color: var(--text-muted);
            box-shadow: none;
            cursor: not-allowed;
            border: 1px solid var(--border-subtle);
        }

        .preset-chip-3d {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 11px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 8px;
        }

        .preset-chip-3d:hover {
            border-color: var(--accent-gold);
            background: var(--bg-surface-elevated);
            transform: translateX(4px);
        }

        /* Split Screen Comparison Slider */
        .split-slider-container {
            position: relative;
            width: 100%;
            height: 240px;
            border-radius: 16px;
            overflow: hidden;
            background: #000000;
            margin-bottom: 24px;
            border: 1px solid var(--border-subtle);
            user-select: none;
        }

        .split-slider-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .split-slider-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 50%;
            height: 100%;
            overflow: hidden;
            border-right: 2px solid var(--accent-gold);
            box-shadow: 2px 0 15px var(--accent-gold-glow);
        }

        .split-slider-overlay img {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            max-width: none;
        }

        .split-handle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--accent-gold);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 800;
            cursor: ew-resize;
            box-shadow: 0 0 15px var(--accent-gold-glow);
            z-index: 10;
        }

        /* Quad Visual Overlays */
        .quad-grid-3d {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 22px;
        }

        @media (max-width: 900px) {
            .quad-grid-3d { grid-template-columns: repeat(2, 1fr); }
        }

        .quad-card-3d {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .quad-card-3d:hover {
            border-color: var(--accent-gold);
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }

        .quad-img-container {
            width: 100%;
            height: 160px;
            border-radius: 12px;
            overflow: hidden;
            background: #000000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .quad-img-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .quad-tag {
            font-size: 11.5px;
            font-weight: 800;
            color: var(--text-secondary);
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* 3D Risk & Mitigation Matrix Section */
        .matrix-section {
            padding: 70px 0;
            border-top: 1px solid var(--border-subtle);
        }

        .risk-table-3d {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 10px;
        }

        .risk-table-3d th {
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            padding: 12px 20px;
            text-align: left;
        }

        .risk-row {
            background: var(--bg-card);
            backdrop-filter: var(--card-glass-blur);
            border-radius: 16px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .risk-row td {
            padding: 18px 20px;
            border-top: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            font-size: 14px;
        }

        .risk-row td:first-child {
            border-left: 1px solid var(--border-subtle);
            border-top-left-radius: 16px;
            border-bottom-left-radius: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .risk-row td:last-child {
            border-right: 1px solid var(--border-subtle);
            border-top-right-radius: 16px;
            border-bottom-right-radius: 16px;
            color: var(--accent-emerald);
            font-weight: 600;
        }

        .risk-row:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-subtle);
        }

        /* Pitch Deck Presentation Modal */
        .pitch-modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.88);
            backdrop-filter: blur(14px);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .pitch-modal-box {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 28px;
            max-width: 960px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 36px;
            position: relative;
            box-shadow: var(--shadow-3d);
        }

        .pitch-nav-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 14px;
            overflow-x: auto;
        }

        .pitch-tab-btn {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 700;
            color: var(--text-secondary);
            cursor: pointer;
            white-space: nowrap;
        }

        .pitch-tab-btn.active {
            background: var(--accent-gold);
            color: #ffffff;
            border-color: var(--accent-gold);
            box-shadow: 0 4px 14px var(--accent-gold-glow);
        }

        .slide-content-pane {
            display: none;
            animation: fadeIn 0.3s ease;
        }

        .slide-content-pane.active {
            display: block;
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

        /* Toast notifications */
        #toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 14px 24px;
            border-radius: 14px;
            background: var(--bg-card);
            border: 1px solid var(--accent-gold);
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 700;
            box-shadow: var(--shadow-3d);
            z-index: 2000;
            display: none;
            animation: slideUp 0.3s ease;
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <!-- Ambient Visual Glow & Grid -->
    <div class="ambient-glow"></div>
    <div class="ambient-glow-secondary"></div>
    <div class="bg-grid"></div>

    <!-- Navigation -->
    <nav>
        <a href="#" class="nav-brand">
            <div class="nav-logo-3d">👁️</div>
            <div class="nav-logo-text">
                OptiNova <span>AI</span>
            </div>
        </a>

        <ul class="nav-links">
            <li><a href="#screening">AI Screening</a></li>
            <li><a href="#how-it-works">Architecture</a></li>
            <li><a href="#depth-layers">3D Retinal Layers</a></li>
            <li><a href="#risks">Risk Mitigation</a></li>
            <li><a href="#telemedicine">Tele-Triage</a></li>
        </ul>

        <div class="nav-actions">
            <button class="btn-pitch-deck" onclick="openPitchModal(0)">
                <span>📑 SIH 2026 Pitch Deck</span>
            </button>
            <button class="theme-toggle-btn" id="themeToggle" onclick="toggleTheme()">
                <span id="themeIcon">🌙</span> <span id="themeLabel">Dark</span>
            </button>
            <a href="#screening" class="btn-pill-primary">Launch Screening</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
        <div class="container">
            <div class="hero-badge-pill">
                <span class="pulse-indicator"></span>
                <span>SMART INDIA HACKATHON 2026 • SIH26038 (TEAM OPTINOVA)</span>
            </div>

            <h1 class="hero-title">
                Eliminate preventable blindness with <span class="highlight-gold">retinal intelligence</span>.
            </h1>

            <p class="hero-sub">
                A MATLAB-native, zero-CAPEX explainable AI screening system providing automated quality gating, multi-class DR severity grading, Grad-CAM heatmaps, and queue routing for rural health clinics.
            </p>

            <div class="hero-actions">
                <a href="#screening" class="btn-hero-3d">
                    <span>✦ Launch AI Screening Studio</span>
                </a>
                <button onclick="selectSample('sample_06_moderate_dr.png')" class="btn-hero-glass">
                    <span>⚡ Load Moderate DR Benchmark</span>
                </button>
                <button onclick="openPitchModal(2)" class="btn-hero-glass">
                    <span>🔬 Technical Approach</span>
                </button>
            </div>

            <!-- Hero Stats Strip (SIH Data Grounding) -->
            <div class="hero-stats-card">
                <div class="stat-column">
                    <div class="stat-num">&lt; 40 ms</div>
                    <div class="stat-subtext">Laplacian Edge QC</div>
                </div>
                <div class="stat-column">
                    <div class="stat-num">&gt; 90%</div>
                    <div class="stat-subtext">Referable Sensitivity</div>
                </div>
                <div class="stat-column">
                    <div class="stat-num">136,875</div>
                    <div class="stat-subtext">Annual District Hub Capacity</div>
                </div>
                <div class="stat-column">
                    <div class="stat-num">&lt; 30 sec</div>
                    <div class="stat-subtext">Remote Doctor Sign-off</div>
                </div>
            </div>
        </div>
    </section>

    <!-- 4-Step Technical Pipeline Section -->
    <section class="how-it-works-section" id="how-it-works">
        <div class="container">
            <div class="section-header">
                <span class="section-pill-tag">Engineering Pipeline</span>
                <h2 class="section-title">End-to-End Technical Architecture</h2>
                <p class="section-desc">From edge Laplacian sharpness gating to sub-second Grad-CAM explainability and bandwidth-resilient telemetry.</p>
            </div>

            <div class="how-grid-4">
                <div class="how-3d-card">
                    <span class="how-card-num">01</span>
                    <div class="how-icon-box-3d">🛡️</div>
                    <h3 class="how-card-title">Edge DSP & QC</h3>
                    <p class="how-card-text">
                        <strong>Laplacian Sharpness Check:</strong> Drops blurred scans locally (<code>Var(∇²I) &lt; τ</code>) in &lt;40 ms before uplink transmission. <strong>CIELAB CLAHE:</strong> Normalizes uneven illumination across diverse fundus scopes.
                    </p>
                </div>

                <div class="how-3d-card">
                    <span class="how-card-num">02</span>
                    <div class="how-icon-box-3d">🔬</div>
                    <h3 class="how-card-title">Vessel & Lesions</h3>
                    <p class="how-card-text">
                        <strong>Frangi Multiscale Filtering:</strong> Extracts vessel tree topology. Green-channel top-hat morphology isolates microaneurysms (&lt;125 µm), dot-blot hemorrhages, and hard lipid exudates.
                    </p>
                </div>

                <div class="how-3d-card">
                    <span class="how-card-num">03</span>
                    <div class="how-icon-box-3d">📊</div>
                    <h3 class="how-card-title">Calibrated Grading</h3>
                    <p class="how-card-text">
                        <strong>Cost-Sensitive Inference:</strong> EfficientNet-B0 tuned via Youden's J-index enforcing &gt;90% sensitivity on Referable DR (Grade ≥2) with Platt-calibrated probability thresholds.
                    </p>
                </div>

                <div class="how-3d-card">
                    <span class="how-card-num">04</span>
                    <div class="how-icon-box-3d">💡</div>
                    <h3 class="how-card-title">XAI & Telemetry</h3>
                    <p class="how-card-text">
                        <strong>Grad-CAM Localization:</strong> Backpropagates target gradients onto lesion clusters for doctor sign-off in &lt;30s. <strong>WebP + SQLite Store-and-Forward:</strong> Shrinks payloads by 96% with offline fault tolerance.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- 3D Retinal Depth Layers Visualization -->
    <section class="retina-3d-section" id="depth-layers">
        <div class="container">
            <div class="depth-card">
                <div class="depth-grid">
                    <div>
                        <span class="section-pill-tag">Spatial Decomposition</span>
                        <h2 class="section-title" style="font-size:30px; margin-bottom:12px;">Interactive 3D Retinal Depth Map</h2>
                        <p class="section-desc" style="font-size:15px; margin-bottom:22px;">
                            Click each anatomical layer to explore how OptiNova's computer vision filters isolate individual biomarkers in 3D retinal space.
                        </p>

                        <div class="depth-controls">
                            <div class="depth-pill active" onclick="activateLayer('base', this)">
                                <div>
                                    <strong style="font-size:14px; color:var(--text-primary);">Layer 1: Fundus CIELAB Canvas</strong>
                                    <p style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Illumination-equalized retinal background & optic disc anchor</p>
                                </div>
                                <span style="font-size:18px;">🔴</span>
                            </div>

                            <div class="depth-pill" onclick="activateLayer('vessels', this)">
                                <div>
                                    <strong style="font-size:14px; color:var(--text-primary);">Layer 2: Frangi 2D Vessel Tree</strong>
                                    <p style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Multiscale directional eigenvalues isolating arteriolar caliber</p>
                                </div>
                                <span style="font-size:18px;">🟡</span>
                            </div>

                            <div class="depth-pill" onclick="activateLayer('lesions', this)">
                                <div>
                                    <strong style="font-size:14px; color:var(--text-primary);">Layer 3: Microaneurysms & Exudates</strong>
                                    <p style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Green top-hat morphology segmenting discrete micro-lesions</p>
                                </div>
                                <span style="font-size:18px;">🩸</span>
                            </div>

                            <div class="depth-pill" onclick="activateLayer('gradcam', this)">
                                <div>
                                    <strong style="font-size:14px; color:var(--text-primary);">Layer 4: Grad-CAM Saliency Field</strong>
                                    <p style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Gradient backpropagation highlighting decisive pathological triggers</p>
                                </div>
                                <span style="font-size:18px;">🔮</span>
                            </div>
                        </div>
                    </div>

                    <div class="retina-spatial-view" id="spatialViewport">
                        <div class="spatial-layer layer-base" id="layerBase"></div>
                        <div class="spatial-layer layer-vessels" id="layerVessels"></div>
                        <div class="spatial-layer layer-lesions" id="layerLesions"></div>
                        <div class="spatial-layer layer-gradcam" id="layerGradcam"></div>
                        <div style="position:absolute; bottom:14px; font-size:11px; color:var(--text-muted); z-index:10; font-family:var(--font-mono);">
                            OptiNova 3D Spatial Retinal Engine • Multi-Spectral
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Screening Studio Workspace -->
    <section class="screening-studio" id="screening">
        <div class="container">
            <div class="studio-card-3d">
                <div class="studio-layout">

                    <!-- Left: Control & Image Input -->
                    <div>
                        <div class="panel-header-sub">
                            <div class="panel-heading-title">
                                <span>📷</span> Edge Image Acquisition
                            </div>
                            <span style="font-size:11px; color:var(--accent-emerald); font-weight:700;">UVC/V4L2 Compatible</span>
                        </div>

                        <div class="drop-zone-3d" id="dropZone" onclick="document.getElementById('fileInput').click()">
                            <div class="drop-zone-icon-3d">📸</div>
                            <div style="font-weight:800; font-size:15px; color:var(--text-primary); margin-bottom:4px;">Upload Fundus Capture</div>
                            <div style="font-size:12px; color:var(--text-muted);">Supports standard PNG, JPG, or DICOM fundus scopes</div>
                            <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFileSelect(event)">
                        </div>
                        <div id="fileSelectionText" style="font-size:12px; color:var(--accent-gold); font-weight:700; margin-top:8px; text-align:center;"></div>

                        <button class="btn-run-pipeline-3d" id="btnRun" onclick="runScreening()" disabled>
                            <span>🚀 Run OptiNova AI Inference</span>
                        </button>

                        <div style="margin-top:24px;">
                            <div style="font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.8px; color:var(--text-muted); margin-bottom:12px; display:flex; justify-content:space-between;">
                                <span>Clinical Benchmark Cases</span>
                                <span>Kaggle APTOS & Messidor-2</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_01_clear.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🟢</span>
                                    <div style="font-size:13px; font-weight:700;">Grade 0: Normal Retina</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(16,185,129,0.15); color:var(--accent-emerald);">Clear</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_02_low_contrast.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🟡</span>
                                    <div style="font-size:13px; font-weight:700;">Low Contrast Image</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(245,158,11,0.15); color:var(--accent-gold);">CLAHE Fix</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_03_blurry.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🔴</span>
                                    <div style="font-size:13px; font-weight:700;">Blurry Scan (&lt;40ms QC)</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(244,63,94,0.15); color:var(--accent-rose);">Drop &lt;τ</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_06_moderate_dr.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🟠</span>
                                    <div style="font-size:13px; font-weight:700;">Grade 2: Moderate DR</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(168,85,247,0.15); color:var(--accent-purple);">Referable</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_07_severe_dr.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🔴</span>
                                    <div style="font-size:13px; font-weight:700;">Grade 3: Severe DR</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(244,63,94,0.15); color:var(--accent-rose);">Hemorrhages</span>
                            </div>

                            <div class="preset-chip-3d" onclick="selectSample('sample_08_proliferative_dr.png')">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <span>🟣</span>
                                    <div style="font-size:13px; font-weight:700;">Grade 4: Proliferative PDR</div>
                                </div>
                                <span style="font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; background:rgba(244,63,94,0.15); color:var(--accent-rose);">Urgent NV</span>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Diagnostic Output -->
                    <div>
                        <div class="panel-header-sub">
                            <div class="panel-heading-title">
                                <span>🔬</span> Multi-Module Diagnostic & Explainability Studio
                            </div>
                            <span style="font-size:12px; color:var(--text-muted);">Real-Time INT8 Inference</span>
                        </div>

                        <!-- Initial Empty State -->
                        <div id="emptyPlaceholder" style="border:1px dashed var(--border-subtle); border-radius:20px; padding:70px 24px; text-align:center; background:var(--bg-surface);">
                            <div style="font-size:42px; margin-bottom:12px;">👁️</div>
                            <h4 style="font-family:var(--font-display); font-size:19px; font-weight:800; margin-bottom:6px; color:var(--text-primary);">Awaiting Retinal Image</h4>
                            <p style="font-size:14px; color:var(--text-secondary); max-width:440px; margin:0 auto;">
                                Upload a fundus scan or select a benchmark preset on the left to execute Modules 1 to 5 with Grad-CAM explainability.
                            </p>
                        </div>

                        <!-- Processing State -->
                        <div id="scanLoader" style="display:none; padding:70px 20px; text-align:center;">
                            <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--text-primary); margin-bottom:8px;">
                                Executing Neural Pipeline...
                            </div>
                            <p style="font-size:13.5px; color:var(--text-muted); font-family:var(--font-mono);">
                                Laplacian QC → CIELAB CLAHE → Frangi Vessel Segmentation → Youden-J Grading → Grad-CAM
                            </p>
                            <div style="display:flex; justify-content:center; gap:10px; margin-top:24px;">
                                <div style="width:12px; height:12px; border-radius:50%; background:var(--accent-gold); animation:pulse-ring 1s infinite;"></div>
                                <div style="width:12px; height:12px; border-radius:50%; background:var(--accent-emerald); animation:pulse-ring 1.2s infinite;"></div>
                                <div style="width:12px; height:12px; border-radius:50%; background:var(--accent-purple); animation:pulse-ring 1.4s infinite;"></div>
                            </div>
                        </div>

                        <!-- Diagnostic Results Container -->
                        <div id="resultsContainer" style="display:none; animation:fadeIn 0.35s ease;">

                            <!-- Severity Banner -->
                            <div id="resBanner" style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:20px; padding:22px 26px; margin-bottom:20px; display:flex; align-items:center; justify-content:space-between; position:relative;">
                                <div>
                                    <h3 id="resGradeTitle" style="font-family:var(--font-display); font-size:23px; font-weight:900; color:var(--text-primary);">Grade 2: Moderate NPDR</h3>
                                    <p id="resConfidence" style="font-size:13px; color:var(--text-secondary); margin-top:4px;">Calibrated Confidence: 91.4% • Platt-Calibrated</p>
                                </div>
                                <span id="resUrgencyBadge" style="padding:8px 18px; border-radius:30px; font-weight:800; font-size:12px; letter-spacing:0.6px; text-transform:uppercase;">REFERRAL REQUIRED</span>
                            </div>

                            <!-- Interactive Split Comparison Slider -->
                            <div class="split-slider-container" id="splitSlider">
                                <img id="splitImgBase" class="split-slider-img" src="" alt="Base Image">
                                <div class="split-slider-overlay" id="splitOverlay">
                                    <img id="splitImgOverlay" src="" alt="Overlay Image">
                                </div>
                                <div class="split-handle" id="splitHandle">↔</div>
                            </div>

                            <!-- 4-Quad Visual Studio -->
                            <div class="quad-grid-3d">
                                <div class="quad-card-3d" onclick="setSplitMode('orig', 'Enhanced CLAHE')">
                                    <div class="quad-img-container">
                                        <img id="imgOrig" src="" alt="Raw">
                                    </div>
                                    <div class="quad-tag">1. Raw Capture</div>
                                </div>

                                <div class="quad-card-3d" onclick="setSplitMode('enhanced', 'CLAHE Contrast')">
                                    <div class="quad-img-container">
                                        <img id="imgEnhanced" src="" alt="Enhanced">
                                    </div>
                                    <div class="quad-tag">2. CLAHE (Mod 1)</div>
                                </div>

                                <div class="quad-card-3d" onclick="setSplitMode('overlay', 'Lesion Masks')">
                                    <div class="quad-img-container">
                                        <img id="imgOverlay" src="" alt="Overlay">
                                    </div>
                                    <div class="quad-tag">3. Lesion Overlay</div>
                                </div>

                                <div class="quad-card-3d" onclick="setSplitMode('gradcam', 'Grad-CAM Saliency')">
                                    <div class="quad-img-container">
                                        <img id="imgGradcam" src="" alt="Grad-CAM">
                                    </div>
                                    <div class="quad-tag">4. Grad-CAM XAI</div>
                                </div>
                            </div>

                            <!-- Biomarker Telemetry Grid -->
                            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-bottom:20px;">
                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Microaneurysms</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--text-primary);" id="bmMAs">0</div>
                                </div>

                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Hard Exudates</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--text-primary);" id="bmExudates">0</div>
                                </div>

                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Hemorrhages</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--text-primary);" id="bmHems">0</div>
                                </div>

                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Laplacian Focus (τ)</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--accent-emerald);" id="bmFocus">0.0</div>
                                </div>

                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Grad-CAM IoU Overlap</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--accent-gold);" id="bmCorrelation">0.00</div>
                                </div>

                                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:14px; padding:12px 16px;">
                                    <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Neovascularization</div>
                                    <div style="font-family:var(--font-display); font-size:22px; font-weight:800; color:var(--accent-purple);" id="bmNV">None</div>
                                </div>
                            </div>

                            <!-- Clinical Rationale Box -->
                            <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:16px; padding:18px; margin-bottom:20px;">
                                <div style="font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:0.8px; color:var(--text-primary); margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                                    <span>📋</span> Explainable AI Clinical Rationale (ICDR Protocol)
                                </div>
                                <div id="resRationaleText" style="font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); line-height:1.6; background:var(--bg-body); padding:12px; border-radius:10px; border:1px solid var(--border-subtle); white-space:pre-wrap;"></div>
                            </div>

                            <!-- Doctor Actions -->
                            <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                                <button onclick="showToast('✓ Doctor Approved in <30s: Case Cleared')" style="padding:11px 18px; border-radius:12px; background:var(--accent-emerald); color:#fff; border:none; font-weight:700; font-size:13px; cursor:pointer;">
                                    ✓ Approve Diagnosis (&lt;30s)
                                </button>
                                <button onclick="showToast('✎ Override Logged: Flagged for Panel Adjudication')" style="padding:11px 18px; border-radius:12px; background:var(--bg-surface); color:var(--text-primary); border:1px solid var(--border-subtle); font-weight:700; font-size:13px; cursor:pointer;">
                                    ✎ Clinical Override
                                </button>
                                <button onclick="showToast('⚑ Case Escalated to Vitreo-Retinal Specialist')" style="padding:11px 18px; border-radius:12px; background:rgba(168,85,247,0.15); color:var(--accent-purple); border:1px solid rgba(168,85,247,0.3); font-weight:700; font-size:13px; cursor:pointer;">
                                    ⚑ Escalate Specialist
                                </button>
                                <button onclick="window.print()" style="margin-left:auto; padding:11px 18px; border-radius:12px; background:var(--bg-surface); color:var(--text-muted); border:1px solid var(--border-subtle); font-weight:700; font-size:13px; cursor:pointer;">
                                    🖨️ Export PDF
                                </button>
                            </div>

                        </div>
                    </div>

                </div>
            </div>
        </div>
    </section>

    <!-- Operational & Clinical Risk Mitigation Table (Slide 4 Details) -->
    <section class="matrix-section" id="risks">
        <div class="container">
            <div class="section-header">
                <span class="section-pill-tag">Clinical & Operational Safeguards</span>
                <h2 class="section-title">Risk & Engineering Mitigation Matrix</h2>
                <p class="section-desc">Audited fail-safes designed specifically for high-volume rural tele-ophthalmology.</p>
            </div>

            <table class="risk-table-3d">
                <thead>
                    <tr>
                        <th>Root Vulnerability</th>
                        <th>Clinical Impact</th>
                        <th>Engineering Technical Mitigation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="risk-row">
                        <td>Diagnostic Leakage (False Negatives)</td>
                        <td>High cost of missing proliferative DR (Grade ≥2) under standard symmetric cross-entropy loss.</td>
                        <td><strong>Youden's J-Index Asymmetric Thresholding:</strong> Biases the ROC operating boundary toward &gt;90% recall.</td>
                    </tr>
                    <tr class="risk-row">
                        <td>Rural Backhaul Jitter & Outages</td>
                        <td>Sub-2 Mbps links, high packet latency, and intermittent cellular dropouts in rural clinics.</td>
                        <td><strong>WebP Quantization + SQLite Store-and-Forward:</strong> Shrinks payload by 96% (&lt;400 KB) with offline caching.</td>
                    </tr>
                    <tr class="risk-row">
                        <td>Cross-Sensor Domain Shift</td>
                        <td>Variations in optical resolution, field of view (FOV), and colour sensor profiles across camera models.</td>
                        <td><strong>CIELAB Contrast Equalization & Heavy Augmentation:</strong> Normalizes luminance via CLAHE; trained across APTOS & Messidor-2.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>

    <!-- Telemedicine Queue Simulator Section (Slide 5 Impact) -->
    <section class="how-it-works-section" id="telemedicine" style="border-top:1px solid var(--border-subtle); padding-top:70px;">
        <div class="container">
            <div class="studio-card-3d">
                <div class="section-header" style="margin-bottom:30px;">
                    <span class="section-pill-tag">Discrete-Event Simulink Validation</span>
                    <h2 class="section-title">136,875 Annual Patient Capacity Proof</h2>
                    <p class="section-desc">Simulate patient queue dynamics and bandwidth optimization over 2 Mbps rural links.</p>
                </div>

                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:36px; align-items:center;">
                    <div>
                        <div style="margin-bottom:20px;">
                            <div style="display:flex; justify-content:space-between; font-size:13.5px; font-weight:700; margin-bottom:8px;">
                                <span>Connected Rural Clinics</span>
                                <span style="color:var(--accent-gold);" id="lblClinics">25 Clinics</span>
                            </div>
                            <input type="range" min="5" max="60" value="25" id="sliderClinics" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                        </div>

                        <div style="margin-bottom:20px;">
                            <div style="display:flex; justify-content:space-between; font-size:13.5px; font-weight:700; margin-bottom:8px;">
                                <span>Ophthalmologists on Shift</span>
                                <span style="color:var(--accent-gold);" id="lblDoctors">4 Doctors</span>
                            </div>
                            <input type="range" min="1" max="10" value="4" id="sliderDoctors" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                        </div>

                        <div style="margin-bottom:20px;">
                            <div style="display:flex; justify-content:space-between; font-size:13.5px; font-weight:700; margin-bottom:8px;">
                                <span>Uplink Bandwidth per Clinic</span>
                                <span style="color:var(--accent-gold);" id="lblBandwidth">2.0 Mbps</span>
                            </div>
                            <input type="range" min="0.5" max="10" step="0.5" value="2.0" id="sliderBandwidth" oninput="updateSim()" style="width:100%; accent-color:var(--accent-gold);">
                        </div>

                        <p style="font-size:13px; color:var(--text-muted); line-height:1.5;">
                            ⚡ <strong>80% Specialist Workload Reduction:</strong> Auto-triage resolves 60% of healthy cases locally, routing only confirmed Referable DR cases (Level 2+) to district ophthalmologists.
                        </p>
                    </div>

                    <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:20px; padding:26px;">
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
                            <div style="background:var(--bg-card); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:800; color:var(--text-muted); text-transform:uppercase;">Annual Patients</div>
                                <div style="font-family:var(--font-display); font-size:26px; font-weight:900; color:var(--text-primary); margin-top:4px;" id="simCapacity">136,875</div>
                            </div>

                            <div style="background:var(--bg-card); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:800; color:var(--text-muted); text-transform:uppercase;">Doctor Utilization</div>
                                <div style="font-family:var(--font-display); font-size:26px; font-weight:900; color:var(--accent-emerald); margin-top:4px;" id="simDoctorUtil">78.2%</div>
                            </div>

                            <div style="background:var(--bg-card); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:800; color:var(--text-muted); text-transform:uppercase;">Avg Triage Latency</div>
                                <div style="font-family:var(--font-display); font-size:26px; font-weight:900; color:var(--accent-gold); margin-top:4px;" id="simWaitTime">3.4 min</div>
                            </div>

                            <div style="background:var(--bg-card); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                                <div style="font-size:11px; font-weight:800; color:var(--text-muted); text-transform:uppercase;">WebP Payload Transmission</div>
                                <div style="font-family:var(--font-display); font-size:26px; font-weight:900; color:var(--accent-purple); margin-top:4px;" id="simUploadDelay">1.6s</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- SIH 2026 Pitch Deck Modal (Slides 1 to 6) -->
    <div class="pitch-modal-backdrop" id="pitchModal" onclick="closePitchModal(event)">
        <div class="pitch-modal-box" onclick="event.stopPropagation()">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <div>
                    <span style="font-size:11px; font-weight:800; color:var(--accent-gold); text-transform:uppercase; letter-spacing:1px;">SMART INDIA HACKATHON 2026</span>
                    <h3 style="font-family:var(--font-display); font-size:22px; font-weight:900; color:var(--text-primary);">Optinova Pitch Deck (SIH26038)</h3>
                </div>
                <button onclick="closePitchModal()" style="background:var(--bg-badge); border:1px solid var(--border-subtle); width:36px; height:36px; border-radius:50%; color:var(--text-primary); cursor:pointer; font-size:18px;">&times;</button>
            </div>

            <div class="pitch-nav-tabs">
                <button class="pitch-tab-btn active" onclick="switchPitchSlide(0, this)">Slide 1: Title & Theme</button>
                <button class="pitch-tab-btn" onclick="switchPitchSlide(1, this)">Slide 2: Objective</button>
                <button class="pitch-tab-btn" onclick="switchPitchSlide(2, this)">Slide 3: Tech Approach</button>
                <button class="pitch-tab-btn" onclick="switchPitchSlide(3, this)">Slide 4: Feasibility & Edge</button>
                <button class="pitch-tab-btn" onclick="switchPitchSlide(4, this)">Slide 5: Impact & Scale</button>
                <button class="pitch-tab-btn" onclick="switchPitchSlide(5, this)">Slide 6: Research & Refs</button>
            </div>

            <!-- Slide 1 -->
            <div class="slide-content-pane active" id="slide0">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:10px;">SMART INDIA HACKATHON 2026</h4>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px;">
                    <div style="background:var(--bg-surface); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                        <strong style="color:var(--accent-gold);">Problem Statement ID:</strong> SIH26038<br>
                        <strong style="color:var(--accent-gold);">Title:</strong> Explainable AI for Diabetic Retinopathy Screening in Rural India
                    </div>
                    <div style="background:var(--bg-surface); padding:16px; border-radius:14px; border:1px solid var(--border-subtle);">
                        <strong style="color:var(--accent-gold);">Theme:</strong> MedTech / Clean & Green technology<br>
                        <strong style="color:var(--accent-gold);">PS Category:</strong> Software | <strong>Team:</strong> Optinova
                    </div>
                </div>
            </div>

            <!-- Slide 2 -->
            <div class="slide-content-pane" id="slide1">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:10px;">Idea Objective</h4>
                <p style="font-size:15px; color:var(--text-secondary); line-height:1.7;">
                    To eliminate preventable blindness in rural India by building a MATLAB-native, explainable AI screening system that provides automated quality gating, multi-class DR severity grading, visual Grad-CAM heatmap telemetry, and optimized telemedicine queue routing for rural health clinics.
                </p>
            </div>

            <!-- Slide 3 -->
            <div class="slide-content-pane" id="slide2">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:14px;">Technical Approach</h4>
                <ul style="color:var(--text-secondary); font-size:14px; line-height:1.8; padding-left:20px;">
                    <li><strong>Laplacian Sharpness Check:</strong> Drops blurred scans locally (<code>Var(∇²I) &lt; τ</code>) in &lt;40 ms before uplink transmission.</li>
                    <li><strong>CIELAB & Vessel Filtering:</strong> CLAHE normalizes illumination; green-channel top-hat morphology isolates lesions.</li>
                    <li><strong>Cost-Sensitive Classification:</strong> EfficientNet-B0 tuned via Youden's J-index enforcing &gt;90% sensitivity on Grade ≥2.</li>
                    <li><strong>Grad-CAM Localization:</strong> Backpropagates gradients for remote doctor verification in &lt;30 seconds.</li>
                    <li><strong>WebP + Offline SQLite Store-and-Forward:</strong> Compresses payload to &lt;400 KB with offline local caching.</li>
                </ul>
            </div>

            <!-- Slide 4 -->
            <div class="slide-content-pane" id="slide3">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:14px;">Feasibility & Zero-CAPEX Edge Runtime</h4>
                <ul style="color:var(--text-secondary); font-size:14px; line-height:1.8; padding-left:20px;">
                    <li><strong>Hardware Compatibility:</strong> Commodity x86 & 64-bit ARM (Intel Core i3 / Raspberry Pi 4 / Android POS).</li>
                    <li><strong>Memory & Footprint:</strong> &lt;1.2 GB peak RAM; model quantized via dynamic INT8 precision for sub-watt edge inference.</li>
                    <li><strong>Zero CAPEX:</strong> Standard UVC/V4L2 camera protocols with legacy non-mydriatic fundus scopes at rural PHCs.</li>
                    <li><strong>Clinical Validation:</strong> Trained on Kaggle APTOS 2019 (3,662 samples), validated on Messidor-2 (1,748 images), EyePACS, and DRIVE.</li>
                </ul>
            </div>

            <!-- Slide 5 -->
            <div class="slide-content-pane" id="slide4">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:14px;">Impact & Community Benefits</h4>
                <ul style="color:var(--text-secondary); font-size:14px; line-height:1.8; padding-left:20px;">
                    <li><strong>Prevents Blindness:</strong> Diagnoses early-stage DR (Levels 1 & 2) directly at rural Primary Health Centres (PHCs).</li>
                    <li><strong>Reduces Expense:</strong> Eliminates non-essential travel costs by resolving 60% of healthy cases locally.</li>
                    <li><strong>80% Specialist Workload Reduction:</strong> Auto-triage routes only confirmed Referable cases (Level 2+) to specialists.</li>
                    <li><strong>136,875 Patients/Year:</strong> Discrete-event Simulink modeling proves district capacity with zero queue backlog.</li>
                </ul>
            </div>

            <!-- Slide 6 -->
            <div class="slide-content-pane" id="slide5">
                <h4 style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:14px;">Research & Clinical References</h4>
                <ul style="color:var(--text-secondary); font-size:14px; line-height:1.8; padding-left:20px;">
                    <li><strong>ICDR Scale:</strong> International Clinical Diabetic Retinopathy Scale (Levels 0–4).</li>
                    <li><strong>Grad-CAM:</strong> Selvaraju, R. R., et al. ICCV 2017.</li>
                    <li><strong>Frangi Filtering:</strong> Frangi, A. F., et al. MICCAI 1998.</li>
                    <li><strong>Datasets:</strong> Kaggle APTOS 2019 (3,662 imgs), Messidor-2 (1,748 imgs), DRIVE Database.</li>
                    <li><strong>Frameworks:</strong> MathWorks MATLAB (Deep Learning gradCAM, adapthisteq, Simulink SimEvents), National Health Portal (NHP) India.</li>
                </ul>
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
                    <div class="nav-logo-3d" style="width:32px; height:32px; font-size:18px;">👁️</div>
                    <span style="font-family:var(--font-display); font-weight:900; font-size:18px;">OptiNova AI</span>
                </div>
                <div style="font-size:13px; color:var(--text-muted);">
                    Smart India Hackathon 2026 (SIH26038) • Team Optinova • Zero-CAPEX Rural Retinal Intelligence
                </div>
                <div style="font-size:12px; color:var(--accent-gold); font-weight:700;">
                    🟢 All 5 Pipeline Modules Active
                </div>
            </div>
        </div>
    </footer>

    <script>
        let selectedFile = null;
        let selectedSampleName = null;
        let lastScreenData = null;

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

        // Initialize Theme from localStorage
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
                document.getElementById('fileSelectionText').innerText = "Selected Scan: " + selectedFile.name;
                document.getElementById('btnRun').disabled = false;
            }
        }

        function selectSample(sampleName) {
            selectedSampleName = sampleName;
            selectedFile = null;
            document.getElementById('fileSelectionText').innerText = "Preset: " + sampleName;
            document.getElementById('btnRun').disabled = false;
            
            const screeningEl = document.getElementById('screening');
            if (screeningEl) {
                screeningEl.scrollIntoView({ behavior: 'smooth' });
            }
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
                    throw new Error("Pipeline Error (" + res.status + "): " + text);
                }
                return res.json();
            })
            .then(data => {
                lastScreenData = data;
                loader.style.display = 'none';
                results.style.display = 'block';

                // Update Severity Banner
                const banner = document.getElementById('resBanner');
                const title = document.getElementById('resGradeTitle');
                const conf = document.getElementById('resConfidence');
                const badge = document.getElementById('resUrgencyBadge');

                title.innerText = data.grade_name;
                conf.innerText = `Confidence Score: ${(data.confidence * 100).toFixed(1)}% • Focus Sharpness: ${data.quality.focus_score.toFixed(1)} (τ)`;

                if (data.status === 'reject') {
                    badge.innerText = "QC GATEKEEPER REJECTED";
                    badge.style.backgroundColor = "var(--accent-rose)";
                    badge.style.color = "#ffffff";
                } else if (data.referable) {
                    badge.innerText = "REFERRAL REQUIRED";
                    badge.style.backgroundColor = data.grade_level >= 3 ? "var(--accent-rose)" : "var(--accent-amber)";
                    badge.style.color = "#ffffff";
                } else {
                    badge.innerText = "ROUTINE / NORMAL";
                    badge.style.backgroundColor = "var(--accent-emerald)";
                    badge.style.color = "#ffffff";
                }

                // Update Images
                document.getElementById('imgOrig').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('imgEnhanced').src = "data:image/jpeg;base64," + data.img_enhanced;
                document.getElementById('imgOverlay').src = "data:image/jpeg;base64," + data.img_overlay;
                document.getElementById('imgGradcam').src = "data:image/jpeg;base64," + data.img_gradcam;

                // Setup Split Comparison Default (Raw vs Enhanced)
                document.getElementById('splitImgBase').src = "data:image/jpeg;base64," + data.img_orig;
                document.getElementById('splitImgOverlay').src = "data:image/jpeg;base64," + data.img_enhanced;

                // Update Biomarkers
                document.getElementById('bmMAs').innerText = data.stats.ma_count || 0;
                document.getElementById('bmExudates').innerText = data.stats.exudate_count || 0;
                document.getElementById('bmHems').innerText = data.stats.hem_count || 0;
                document.getElementById('bmFocus').innerText = data.quality.focus_score.toFixed(1);
                document.getElementById('bmCorrelation').innerText = data.correlation_score.toFixed(2);
                document.getElementById('bmNV').innerText = data.stats.nv_flag ? "YES (Active)" : "None";

                // Update Rationale
                document.getElementById('resRationaleText').innerText = data.rationale;
            })
            .catch(err => {
                loader.style.display = 'none';
                alert("Pipeline Execution Error: " + err.message);
            });
        }

        // Split Comparison Slider Logic
        function setSplitMode(type, label) {
            if (!lastScreenData) return;
            const baseImg = document.getElementById('splitImgBase');
            const overlayImg = document.getElementById('splitImgOverlay');

            baseImg.src = "data:image/jpeg;base64," + lastScreenData.img_orig;
            if (type === 'enhanced') overlayImg.src = "data:image/jpeg;base64," + lastScreenData.img_enhanced;
            else if (type === 'overlay') overlayImg.src = "data:image/jpeg;base64," + lastScreenData.img_overlay;
            else if (type === 'gradcam') overlayImg.src = "data:image/jpeg;base64," + lastScreenData.img_gradcam;
            else overlayImg.src = "data:image/jpeg;base64," + lastScreenData.img_enhanced;

            showToast("Comparison Mode: Raw Fundus vs " + label);
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

        // 3D Depth Layer Activation
        function activateLayer(layerType, btn) {
            document.querySelectorAll('.depth-pill').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');

            const base = document.getElementById('layerBase');
            const vessels = document.getElementById('layerVessels');
            const lesions = document.getElementById('layerLesions');
            const gradcam = document.getElementById('layerGradcam');

            if (layerType === 'base') {
                base.style.transform = "translateZ(60px) rotateX(15deg) scale(1.05)";
                vessels.style.transform = "translateZ(20px) rotateX(15deg) opacity(0.4)";
                lesions.style.transform = "translateZ(0px) rotateX(15deg) opacity(0.3)";
                gradcam.style.transform = "translateZ(-20px) rotateX(15deg) opacity(0.2)";
            } else if (layerType === 'vessels') {
                base.style.transform = "translateZ(0px) rotateX(15deg)";
                vessels.style.transform = "translateZ(80px) rotateX(15deg) scale(1.08)";
                lesions.style.transform = "translateZ(40px) rotateX(15deg)";
                gradcam.style.transform = "translateZ(20px) rotateX(15deg)";
            } else if (layerType === 'lesions') {
                base.style.transform = "translateZ(-20px) rotateX(15deg)";
                vessels.style.transform = "translateZ(20px) rotateX(15deg)";
                lesions.style.transform = "translateZ(90px) rotateX(15deg) scale(1.1)";
                gradcam.style.transform = "translateZ(40px) rotateX(15deg)";
            } else if (layerType === 'gradcam') {
                base.style.transform = "translateZ(-40px) rotateX(15deg)";
                vessels.style.transform = "translateZ(0px) rotateX(15deg)";
                lesions.style.transform = "translateZ(40px) rotateX(15deg)";
                gradcam.style.transform = "translateZ(110px) rotateX(15deg) scale(1.12)";
            }
        }

        // Pitch Modal Logic
        function openPitchModal(slideIdx=0) {
            document.getElementById('pitchModal').style.display = 'flex';
            const tabs = document.querySelectorAll('.pitch-tab-btn');
            if (tabs[slideIdx]) switchPitchSlide(slideIdx, tabs[slideIdx]);
        }

        function closePitchModal(e) {
            document.getElementById('pitchModal').style.display = 'none';
        }

        function switchPitchSlide(idx, btn) {
            document.querySelectorAll('.pitch-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.slide-content-pane').forEach((p, i) => {
                p.classList.toggle('active', i === idx);
            });
        }

        // Telemedicine Simulation
        function updateSim() {
            const clinics = parseInt(document.getElementById('sliderClinics').value);
            const doctors = parseInt(document.getElementById('sliderDoctors').value);
            const bw = parseFloat(document.getElementById('sliderBandwidth').value);

            document.getElementById('lblClinics').innerText = clinics + " Clinics";
            document.getElementById('lblDoctors').innerText = doctors + " Doctors";
            document.getElementById('lblBandwidth').innerText = bw.toFixed(1) + " Mbps";

            const annualCap = clinics * 15 * 365;
            document.getElementById('simCapacity').innerText = annualCap.toLocaleString();

            const uploadDelay = (0.4 / (bw / 8.0)).toFixed(1); // 400 KB WebP payload
            document.getElementById('simUploadDelay').innerText = uploadDelay + "s";

            const referablePerDay = clinics * 15 * 0.4;
            const doctorCapacityPerDay = doctors * (8 * 60 / 0.5);
            const util = Math.min(99.5, (referablePerDay / doctorCapacityPerDay) * 100);
            document.getElementById('simDoctorUtil').innerText = util.toFixed(1) + "%";

            const avgWait = (Math.max(0.3, (util / 100) * 3.8) + (uploadDelay / 60)).toFixed(1);
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
            rationale = f"[QUALITY GATEKEEPER REJECTED]\\nReason: {reason}\\nAction: Laplacian focus Var(∇²I) < τ. Please adjust illumination/focus and recapture."
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
