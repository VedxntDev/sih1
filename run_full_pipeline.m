% RUN_FULL_PIPELINE Master MATLAB Demonstration Script for SIH 2026
%
% Project: Explainable AI for DR Screening in Rural India (SIH 2026, PS ID 26038, MathWorks)
% Runs Modules 1 to 5 end-to-end on sample retinal fundus images.

clc; clear; close all;

fprintf('========================================================================\n');
fprintf(' EXPLAINABLE AI FOR DIABETIC RETINOPATHY SCREENING IN RURAL INDIA (SIH 2026)\n');
fprintf(' MathWorks Problem Statement ID: 26038 | Team Working Prototype\n');
fprintf('========================================================================\n\n');

sampleDir = 'data/sample_images';
sampleFiles = {'sample_01_clear.png', 'sample_02_low_contrast.png', 'sample_03_blurry.png', ...
               'sample_06_moderate_dr.png', 'sample_07_severe_dr.png', 'sample_08_proliferative_dr.png'};

for i = 1:length(sampleFiles)
    imgName = sampleFiles{i};
    imgPath = fullfile(sampleDir, imgName);
    if ~exist(imgPath, 'file')
        continue;
    end
    
    fprintf('>>> Processing Image %d/%d: %s\n', i, length(sampleFiles), imgName);
    rawImg = imread(imgPath);
    
    % --- MODULE 1: Quality Gatekeeper & CLAHE Enhancement ---
    [status, enhancedImg, qReport, rejectionReason] = assessAndEnhance(rawImg);
    fprintf('  [Module 1 Quality] Status: %s | Focus: %.1f | FOV: %.1f%% | Contrast: %.1f\n', ...
        upper(status), qReport.focusScore, qReport.fovRatio*100, qReport.contrastScore);
    
    if strcmp(status, 'reject')
        fprintf('  [Module 1 Rejection Action]: %s\n\n', rejectionReason);
        continue;
    end
    
    % --- MODULE 2: Retinal Structure & Lesion Segmentation ---
    [lesionOverlay, lesionStats, structMasks] = segmentRetinalStructures(enhancedImg);
    fprintf('  [Module 2 Lesions] MAs: %d | Exudates: %d | Hemorrhages: %d | NV: %s\n', ...
        lesionStats.maCount, lesionStats.exudateCount, lesionStats.hemCount, string(lesionStats.nvFlag));
    
    % --- MODULE 3: DR Severity Grading & Referable Cutoff ---
    [severityLevel, referableFlag, confidence, classProbs] = gradeDR(lesionStats);
    fprintf('  [Module 3 Severity] Grade %d | Referable: %s | Calibrated Conf: %.1f%%\n', ...
        severityLevel, string(referableFlag), confidence*100);
    
    % --- MODULE 4: Clinical Explainability & Grad-CAM ---
    [gradcamHeatmap, correlationScore, patientReport] = explainPrediction(enhancedImg, severityLevel, referableFlag, confidence, lesionStats, structMasks);
    fprintf('  [Module 4 Explainability] Grad-CAM Lesion Overlap Score: %.2f\n', correlationScore);
    fprintf('  [Clinical Rationale]: %s\n\n', patientReport.rationaleText);
end

% --- MODULE 5: Simulink Telemedicine Workflow Simulation ---
fprintf('>>> [Module 5 Simulink Workflow] Launching Telemedicine Queue Setup...\n');
setup_simulink_queue;
fprintf('========================================================================\n');
fprintf(' END-TO-END PIPELINE DEMONSTRATION COMPLETE!\n');
fprintf('========================================================================\n');
