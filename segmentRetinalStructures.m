function [lesionOverlay, lesionStats, structMasks] = segmentRetinalStructures(img)
% SEGMENTRETINALSTRUCTURES Retinal Structure & DR Lesion Segmentation Engine
%
% Project: Explainable AI for DR Screening in Rural India (SIH 2026, PS ID 26038)
% Module 2: Retinal Structure Segmentation & Lesion Detection
%
% Inputs:
%   img - Enhanced RGB fundus image (uint8, HxWx3)
%
% Outputs:
%   lesionOverlay - RGB image with color-coded structure & lesion overlays
%   lesionStats   - Struct containing counts, areas, vessel density, tortuosity
%   structMasks   - Struct containing individual binary masks for structures & lesions

    if isfloat(img)
        img = uint8(img * 255);
    end

    [height, width, ~] = size(img);
    rChan = double(img(:,:,1));
    gChan = double(img(:,:,2));
    bChan = double(img(:,:,3));

    % 1. Retinal FOV Mask
    grayImg = 0.2989 * rChan + 0.5870 * gChan + 0.1140 * bChan;
    fovMask = (grayImg > 15);
    fovMask = imfill(fovMask, 'holes');

    % 2. Optic Disc (OD) Localization & Segmentation
    % OD is the brightest, smooth circular/elliptical structure
    brightMap = (0.5 * rChan + 0.5 * gChan) .* double(fovMask);
    gaussianOD = imfilter(brightMap, fspecial('gaussian', [31 31], 8), 'replicate');
    [~, maxIdx] = max(gaussianOD(:));
    [odY_center, odX_center] = ind2sub([height, width], maxIdx);

    % Morphological region growing around OD center
    odThresh = gaussianOD > (0.85 * max(gaussianOD(:)));
    odConn = bwlabel(odThresh);
    odLabelAtCenter = odConn(odY_center, odX_center);
    if odLabelAtCenter > 0
        odMask = (odConn == odLabelAtCenter);
    else
        % Fallback circle
        [cX, cY] = meshgrid(1:width, 1:height);
        odMask = sqrt((cX - odX_center).^2 + (cY - odY_center).^2) <= (0.07 * min(height, width));
    end
    odMask = imfill(odMask, 'holes');
    odRadius = sqrt(sum(odMask(:)) / pi);

    % 3. Fovea Localization (Standard geometric offset heuristic)
    % Fovea is ~2.5 Disc Diameters (5 radii) temporal & slightly inferior to OD
    % Determine if OD is on Left (Nasal OS) or Right (Nasal OD) side of frame
    if odX_center < width / 2
        % OD on left -> Fovea to the right (temporal)
        foveaX = min(width - 20, round(odX_center + 4.5 * odRadius));
    else
        % OD on right -> Fovea to the left (temporal)
        foveaX = max(20, round(odX_center - 4.5 * odRadius));
    end
    foveaY = min(height - 20, max(20, round(odY_center + 0.3 * odRadius)));

    [cX, cY] = meshgrid(1:width, 1:height);
    foveaMask = sqrt((cX - foveaX).^2 + (cY - foveaY).^2) <= (0.8 * odRadius);

    % 4. Blood Vessel Extraction (Frangi Vesselness & Matched Filter)
    gNorm = gChan ./ (max(gChan(:)) + 1e-5);
    gInverted = 1.0 - gNorm; % Vessels are dark in green channel, so bright when inverted

    % Morphological background homogenization
    bgEstimate = imfilter(gInverted, fspecial('gaussian', [41 41], 10), 'replicate');
    vesselEnhanced = max(0, gInverted - bgEstimate);

    % Multi-scale matched vessel filtering
    vesselResponse = zeros(height, width);
    angles = 0:15:165;
    for sigma = [1.5, 2.5]
        for theta = angles
            kernel = fspecial('gaussian', [15 15], sigma);
            % Directional derivative along theta
            rad = deg2rad(theta);
            [kx, ky] = meshgrid(-7:7, -7:7);
            rotK = kx * cos(rad) + ky * sin(rad);
            dirKernel = -rotK .* kernel;
            response = imfilter(vesselEnhanced, dirKernel, 'replicate');
            vesselResponse = max(vesselResponse, response);
        end
    end
    
    vesselThreshold = mean(vesselResponse(fovMask)) + 0.8 * std(vesselResponse(fovMask));
    vesselMask = (vesselResponse > vesselThreshold) & fovMask & ~odMask;
    vesselMask = bwareaopen(vesselMask, 15); % Remove tiny noise spots

    % Vessel Density & Tortuosity
    vesselArea = sum(vesselMask(:));
    vesselDensity = vesselArea / (sum(fovMask(:)) + 1e-5);

    % Skeletonize for tortuosity estimation
    vesselSktd = bwskel(vesselMask);
    vesselLength = sum(vesselSktd(:));
    vesselTortuosity = vesselLength / (vesselArea + 1e-5);

    % 5. Microaneurysm (MA) Detection
    % Morphological Top-Hat filtering for small dark circular lesions
    seMA = strel('disk', 4);
    topHat = imtophat(255 - uint8(gChan), seMA);
    topHat(~fovMask | vesselMask | odMask) = 0;
    
    maThresh = mean(topHat(fovMask)) + 2.5 * std(double(topHat(fovMask)));
    maCandidates = (topHat > maThresh);
    
    % Candidate shape and size filtering (MA diameter < 125 um, small area)
    maCC = bwconncomp(maCandidates);
    maStatsProps = regionprops(maCC, 'Area', 'Eccentricity', 'Circularity', 'Centroid');
    maMask = false(height, width);
    maCount = 0;
    for k = 1:maCC.NumObjects
        area = maStatsProps(k).Area;
        ecc = maStatsProps(k).Eccentricity;
        if area >= 2 && area <= 45 && ecc < 0.85
            maMask(maCC.PixelIdxList{k}) = true;
            maCount = maCount + 1;
        end
    end

    % 6. Exudate Segmentation
    % Bright yellow/white waxy lesions outside Optic Disc
    retinaMeanG = mean(gChan(fovMask & ~odMask));
    retinaStdG  = std(gChan(fovMask & ~odMask));
    
    exudateCandidates = (gChan > (retinaMeanG + 1.8 * retinaStdG)) & ...
                        (rChan > (gChan * 0.85)) & ...
                        fovMask & ~odMask;
    
    exudateCandidates = bwareaopen(exudateCandidates, 5);
    exudateMask = imclose(exudateCandidates, strel('disk', 2));
    exudateArea = sum(exudateMask(:));
    exudateCC = bwconncomp(exudateMask);
    exudateCount = exudateCC.NumObjects;

    % 7. Hemorrhage Classification
    % Dark irregular blotches distinct from vessels and MAs
    darkRegions = (gChan < (retinaMeanG - 1.5 * retinaStdG)) & fovMask & ~vesselMask & ~odMask;
    hemCC = bwconncomp(darkRegions);
    hemProps = regionprops(hemCC, 'Area', 'Eccentricity');
    hemMask = false(height, width);
    hemCount = 0;
    for k = 1:hemCC.NumObjects
        area = hemProps(k).Area;
        if area >= 20 % Hemorrhages are larger than MAs
            hemMask(hemCC.PixelIdxList{k}) = true;
            hemCount = hemCount + 1;
        end
    end
    hemorrhageArea = sum(hemMask(:));

    % 8. Neovascularization (NV) Detection
    % Irregular fine vessel proliferation near Optic Disc margin (within 1.5 OD radii)
    odMarginZone = sqrt((cX - odX_center).^2 + (cY - odY_center).^2) <= (2.2 * odRadius) & ~odMask;
    fineVesselsNearOD = vesselMask & odMarginZone;
    nvArea = sum(fineVesselsNearOD(:));
    nvFlag = (nvArea > (0.15 * sum(odMarginZone(:))));

    % Compile Structure Masks
    structMasks = struct();
    structMasks.fovMask = fovMask;
    structMasks.odMask = odMask;
    structMasks.foveaMask = foveaMask;
    structMasks.vesselMask = vesselMask;
    structMasks.maMask = maMask;
    structMasks.exudateMask = exudateMask;
    structMasks.hemMask = hemMask;
    structMasks.fineVesselsNearOD = fineVesselsNearOD;

    % Compile Lesion Statistics Struct
    lesionStats = struct();
    lesionStats.odCenter = [odX_center, odY_center];
    lesionStats.foveaCenter = [foveaX, foveaY];
    lesionStats.vesselDensity = vesselDensity;
    lesionStats.vesselTortuosity = vesselTortuosity;
    lesionStats.maCount = maCount;
    lesionStats.maArea = sum(maMask(:));
    lesionStats.exudateCount = exudateCount;
    lesionStats.exudateArea = exudateArea;
    lesionStats.hemCount = hemCount;
    lesionStats.hemArea = hemorrhageArea;
    lesionStats.nvFlag = nvFlag;
    lesionStats.nvArea = nvArea;

    % 9. Build RGB Overlay Image
    overlay = img;
    % Vessels -> Green (0, 255, 0)
    for c = 1:3
        ch = overlay(:,:,c);
        if c == 2, ch(vesselMask) = 255; else, ch(vesselMask) = 0; end
        overlay(:,:,c) = ch;
    end
    
    % Optic Disc Contour -> Yellow (255, 255, 0)
    odPerim = bwperim(odMask);
    overlay(:,:,1) = uint8(double(overlay(:,:,1)) .* ~odPerim + 255 * double(odPerim));
    overlay(:,:,2) = uint8(double(overlay(:,:,2)) .* ~odPerim + 255 * double(odPerim));
    overlay(:,:,3) = uint8(double(overlay(:,:,3)) .* ~odPerim);

    % Fovea Circle -> Blue (0, 120, 255)
    foveaPerim = bwperim(foveaMask);
    overlay(:,:,1) = uint8(double(overlay(:,:,1)) .* ~foveaPerim);
    overlay(:,:,2) = uint8(double(overlay(:,:,2)) .* ~foveaPerim + 120 * double(foveaPerim));
    overlay(:,:,3) = uint8(double(overlay(:,:,3)) .* ~foveaPerim + 255 * double(foveaPerim));

    % Microaneurysms -> Red Dots (255, 0, 0)
    maDilated = imdilate(maMask, strel('disk', 2));
    overlay(:,:,1) = uint8(double(overlay(:,:,1)) .* ~maDilated + 255 * double(maDilated));
    overlay(:,:,2) = uint8(double(overlay(:,:,2)) .* ~maDilated);
    overlay(:,:,3) = uint8(double(overlay(:,:,3)) .* ~maDilated);

    % Exudates -> Cyan (0, 255, 255)
    overlay(:,:,1) = uint8(double(overlay(:,:,1)) .* ~exudateMask);
    overlay(:,:,2) = uint8(double(overlay(:,:,2)) .* ~exudateMask + 255 * double(exudateMask));
    overlay(:,:,3) = uint8(double(overlay(:,:,3)) .* ~exudateMask + 255 * double(exudateMask));

    % Hemorrhages -> Magenta (255, 0, 255)
    overlay(:,:,1) = uint8(double(overlay(:,:,1)) .* ~hemMask + 255 * double(hemMask));
    overlay(:,:,2) = uint8(double(overlay(:,:,2)) .* ~hemMask);
    overlay(:,:,3) = uint8(double(overlay(:,:,3)) .* ~hemMask + 255 * double(hemMask));

    lesionOverlay = overlay;
end
