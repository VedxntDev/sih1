function [status, enhancedImage, qualityReport, rejectionReason] = assessAndEnhance(img, params)
% ASSESSANDENHANCE Automated Image Quality Gatekeeper and CLAHE Enhancer
%
% Project: Explainable AI for DR Screening in Rural India (SIH 2026, PS ID 26038)
% Module 1: Image Quality Assessment & Enhancement
%
% Inputs:
%   img - Input RGB fundus image (uint8 or double, size HxWx3)
%   params - (Optional) struct specifying custom threshold parameters
%
% Outputs:
%   status          - 'pass' | 'enhance' | 'reject'
%   enhancedImage   - Quality-enhanced RGB image (uint8)
%   qualityReport   - Struct containing numeric quality metrics
%   rejectionReason - Actionable rejection rationale string (or empty)

    if nargin < 2 || isempty(params)
        params = struct();
    end

    % Default justified clinical quality thresholds
    if ~isfield(params, 'focusMinPass'),       params.focusMinPass = 110.0;  end
    if ~isfield(params, 'focusMinReject'),     params.focusMinReject = 45.0;  end
    if ~isfield(params, 'fovMinPass'),         params.fovMinPass = 0.70;    end
    if ~isfield(params, 'fovMinReject'),       params.fovMinReject = 0.55;  end
    if ~isfield(params, 'contrastMinPass'),    params.contrastMinPass = 32.0;  end
    if ~isfield(params, 'contrastMinReject'),  params.contrastMinReject = 18.0; end
    if ~isfield(params, 'illumStdMaxPass'),    params.illumStdMaxPass = 0.16;  end

    % Standardize image data type to uint8
    if isfloat(img)
        if max(img(:)) <= 1.0
            img = uint8(img * 255);
        else
            img = uint8(img);
        end
    end

    [height, width, channels] = size(img);
    if channels == 1
        img = cat(3, img, img, img);
    end

    % Extract channels & green channel (highest contrast for retinal structures)
    rChan = double(img(:,:,1));
    gChan = double(img(:,:,2));
    bChan = double(img(:,:,3));
    grayImg = 0.2989 * rChan + 0.5870 * gChan + 0.1140 * bChan;

    % 1. Field of View (FOV) Mask & Coverage Computation
    fovMask = (grayImg > 15);
    fovMask = imfill(fovMask, 'holes');
    fovArea = sum(fovMask(:));
    totalArea = height * width;
    fovRatio = fovArea / totalArea;

    % 2. Focus / Sharpness Metric (Laplacian Variance & FFT High-Freq Ratio)
    lapKernel = [0 1 0; 1 -4 1; 0 1 0];
    lapFiltered = filter2(lapKernel, gChan);
    if fovArea > 0
        focusScore = var(lapFiltered(fovMask));
    else
        focusScore = var(lapFiltered(:));
    end

    % 2D FFT spectral high-frequency ratio
    fft2D = abs(fftshift(fft2(gChan)));
    [cX, cY] = meshgrid(1:width, 1:height);
    distFromCenter = sqrt((cX - width/2).^2 + (cY - height/2).^2);
    highFreqMask = distFromCenter > (min(height, width) * 0.25);
    fftFocusScore = sum(fft2D(highFreqMask)) / (sum(fft2D(:)) + 1e-6);

    % 3. Illumination Uniformity (Quadrant Mean Brightness Variation)
    halfH = floor(height/2);
    halfW = floor(width/2);
    q1 = gChan(1:halfH, 1:halfW);          q1_mask = fovMask(1:halfH, 1:halfW);
    q2 = gChan(1:halfH, halfW+1:end);      q2_mask = fovMask(1:halfH, halfW+1:end);
    q3 = gChan(halfH+1:end, 1:halfW);      q3_mask = fovMask(halfH+1:end, 1:halfW);
    q4 = gChan(halfH+1:end, halfW+1:end);  q4_mask = fovMask(halfH+1:end, halfW+1:end);

    qMeans = [mean(q1(q1_mask > 0)), mean(q2(q2_mask > 0)), ...
              mean(q3(q3_mask > 0)), mean(q4(q4_mask > 0))];
    qMeans(isnan(qMeans)) = mean(gChan(fovMask));

    illuminationStd = std(qMeans) / (mean(qMeans) + 1e-6);

    % 4. Contrast Score (RMS contrast on green channel inside FOV)
    if fovArea > 0
        contrastScore = std(gChan(fovMask));
        meanBrightness = mean(gChan(fovMask));
    else
        contrastScore = std(gChan(:));
        meanBrightness = mean(gChan(:));
    end

    % Compile Quality Report Struct
    qualityReport = struct();
    qualityReport.focusScore = focusScore;
    qualityReport.fftFocusScore = fftFocusScore;
    qualityReport.fovRatio = fovRatio;
    qualityReport.contrastScore = contrastScore;
    qualityReport.meanBrightness = meanBrightness;
    qualityReport.illuminationStd = illuminationStd;

    % 5. Gatekeeping Decision Logic
    rejectionReason = '';
    if fovRatio < params.fovMinReject
        status = 'reject';
        rejectionReason = sprintf('Incomplete Field of View (Coverage: %.1f%%, Required: >%.1f%%) — Re-align fundus camera centered on pupil.', fovRatio*100, params.fovMinReject*100);
    elseif focusScore < params.focusMinReject
        status = 'reject';
        rejectionReason = sprintf('Out of Focus / Severe Blur (Focus Score: %.1f, Min: %.1f) — Adjust camera focus dial before recapture.', focusScore, params.focusMinReject);
    elseif contrastScore < params.contrastMinReject || meanBrightness < 20.0
        status = 'reject';
        rejectionReason = sprintf('Severe Illumination Deficiency (Contrast: %.1f, Brightness: %.1f) — Increase flash illumination.', contrastScore, meanBrightness);
    elseif focusScore < params.focusMinPass || contrastScore < params.contrastMinPass || illuminationStd > params.illumStdMaxPass
        status = 'enhance';
    else
        status = 'pass';
    end

    % 6. Enhancement Pipeline (for 'enhance' or 'pass' images)
    if strcmp(status, 'reject')
        enhancedImage = img; % Return original if rejected
    else
        % A. CLAHE on green channel (Contrast-Limited Adaptive Histogram Equalization)
        gNorm = uint8(gChan);
        gClahe = adapthisteq(gNorm, 'ClipLimit', 0.02, 'Distribution', 'uniform', 'NumTiles', [8 8]);

        % B. Illumination Normalization (Gaussian Background Subtraction)
        gaussianBlur = imfilter(double(gClahe), fspecial('gaussian', [61 61], 15), 'replicate');
        targetMean = mean(gChan(fovMask));
        gIllumNorm = (double(gClahe) ./ (gaussianBlur + 1e-5)) * targetMean;
        gIllumNorm = uint8(min(255, max(0, gIllumNorm)));

        % C. Denoising via Median Filter
        gDenoised = medfilt2(gIllumNorm, [3 3]);

        % D. Color Re-synthesis (Blend enhanced luminance back to RGB)
        hsvImg = rgb2hsv(img);
        hsvImg(:,:,2) = min(1.0, hsvImg(:,:,2) * 1.15); % Slight saturation boost
        hsvImg(:,:,3) = double(gDenoised) / 255.0;      % Replace Value channel with enhanced green
        enhancedImage = uint8(hsv2rgb(hsvImg) * 255.0);

        % Zero out background outside FOV
        for c = 1:3
            tempC = enhancedImage(:,:,c);
            tempC(~fovMask) = 0;
            enhancedImage(:,:,c) = tempC;
        end
    end
end
