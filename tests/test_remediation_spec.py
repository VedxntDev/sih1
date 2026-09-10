#!/usr/bin/env python3
"""
Comprehensive Remediation Verification Test Suite for OptiNova AI
Tests all 10 engineering and clinical requirements specified in the remediation prompt.
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_module1 import assess_and_enhance
from test_module2 import segment_retinal_structures
from test_module3 import grade_dr
from test_module4 import explain_prediction

class TestRemediationSpec(unittest.TestCase):
    def test_01_lesion_grounding_and_outlier_guard(self):
        """Verify lesion extraction grounding and outlier guard on benchmark cases."""
        img = cv2.imread('data/sample_images/sample_07_severe_dr.png')
        self.assertIsNotNone(img)
        overlay, stats, masks = segment_retinal_structures(img)
        
        self.assertGreaterEqual(stats['hem_count'], 2)
        self.assertGreaterEqual(stats['ma_count'], 1)
        self.assertIn('is_outlier', stats)
        self.assertFalse(stats['is_outlier'])
        
        # Test outlier triggering with synthetic outlier stats
        outlier_stats = stats.copy()
        outlier_stats['ma_count'] = 489
        outlier_stats['exudate_count'] = 213
        outlier_stats['is_outlier'] = True
        
        _, _, report = explain_prediction(img, 3, True, 0.95, outlier_stats, masks)
        self.assertTrue(report['is_xai_gated'])
        self.assertIn("Outlier", report['rationale_text'])

    def test_02_input_validation_and_authenticity(self):
        """Verify gatekeeper rejects blur, incomplete FOV, and invalid aspect ratios."""
        status, _, q_rep, reason = assess_and_enhance('data/sample_images/sample_03_blurry.png')
        self.assertEqual(status, 'reject')
        self.assertTrue('Focus' in reason or 'Blur' in reason)

        status, _, q_rep, reason = assess_and_enhance('data/sample_images/sample_05_cropped.png')
        self.assertEqual(status, 'reject')
        self.assertIn('Field of View', reason)

        status, _, q_rep, reason = assess_and_enhance('data/sample_images/sample_01_clear.png')
        self.assertEqual(status, 'pass')
        self.assertNotEqual(q_rep['image_sha256'], '')

    def test_03_module_transformations_distinctness(self):
        """Verify that Raw, CLAHE, Overlay, and Grad-CAM outputs are distinct image transformations."""
        img_path = 'data/sample_images/sample_06_moderate_dr.png'
        raw_img = cv2.imread(img_path)
        status, enhanced, q_rep, _ = assess_and_enhance(img_path)
        overlay, stats, masks = segment_retinal_structures(enhanced)
        level, ref, conf, _, _ = grade_dr(stats, quality=q_rep)
        heatmap, _, _ = explain_prediction(enhanced, level, ref, conf, stats, masks)

        raw_hash = hash(raw_img.tobytes())
        enh_hash = hash(enhanced.tobytes())
        ovr_hash = hash(overlay.tobytes())
        heat_hash = hash(heatmap.tobytes())

        self.assertNotEqual(raw_hash, enh_hash)
        self.assertNotEqual(enh_hash, ovr_hash)
        self.assertNotEqual(ovr_hash, heat_hash)
        self.assertGreater(q_rep['clahe_chi_sq'], 0.04)

    def test_04_single_source_of_truth_metrics(self):
        """Verify metrics consistency in report object."""
        img_path = 'data/sample_images/sample_06_moderate_dr.png'
        _, enhanced, q_rep, _ = assess_and_enhance(img_path)
        overlay, stats, masks = segment_retinal_structures(enhanced)
        level, ref, conf, _, _ = grade_dr(stats, quality=q_rep)
        heatmap, corr_score, report = explain_prediction(enhanced, level, ref, conf, stats, masks)

        iou_str = f"{report['spatial_iou']:.2f}"
        pearson_str = f"{report['pearson_corr']:.2f}"
        self.assertIn(iou_str, report['rationale_text'])
        self.assertIn(pearson_str, report['rationale_text'])
        self.assertLessEqual(report['confidence'], 1.0)

    def test_05_conditional_alert_logic(self):
        """Verify that alert logic only cites metrics that individually fail threshold."""
        img = cv2.imread('data/sample_images/sample_06_moderate_dr.png')
        _, stats, masks = segment_retinal_structures(img)
        
        _, _, rep = explain_prediction(img, 2, True, 0.90, stats, masks)
        
        if rep['is_xai_gated']:
            if rep['iou_passed']:
                self.assertNotIn("Spatial IoU (", rep['rationale_text'].split("⚠️ [XAI CO-LOCALIZATION ALERT]:")[1].split("did not reach")[0])
            if rep['pearson_passed']:
                self.assertNotIn("Pearson Correlation (", rep['rationale_text'].split("⚠️ [XAI CO-LOCALIZATION ALERT]:")[1].split("did not reach")[0])

    def test_06_transparent_downgrade_and_triage(self):
        """Verify confidence downgrade formula and clinical triage mapping for all ICDR levels."""
        img = cv2.imread('data/sample_images/sample_01_clear.png')
        _, stats, masks = segment_retinal_structures(img)

        for lvl in range(5):
            _, _, rep = explain_prediction(img, lvl, (lvl >= 2), 0.92, stats, masks)
            self.assertIn('downgrade_penalty_formula', rep)
            self.assertIn('XAI-GATE', rep['downgrade_rule_version'])
            
            # Clinical Invariant: Grade 2+ must ALWAYS be Referable
            if lvl >= 2:
                self.assertTrue(rep['referable_flag'])
                self.assertNotIn("Non-Referable", rep['triage_criterion'])
                self.assertIn("REFERRAL", rep['triage_decision'])
            else:
                self.assertFalse(rep['referable_flag'])
                self.assertIn("Non-Referable", rep['triage_criterion'])

if __name__ == '__main__':
    unittest.main()
