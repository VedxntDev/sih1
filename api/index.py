#!/usr/bin/env python3
"""
Vercel Serverless Entry Point for Flask Web App
"""

import sys
import os

# Add root directory to sys.path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set Matplotlib cache dir to /tmp for read-only Vercel serverless environment
os.environ['MPLCONFIGDIR'] = '/tmp'

from app import app

# Expose app instance for Vercel Serverless WSGI
handler = app
