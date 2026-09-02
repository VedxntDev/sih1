# Live Presentation & Demonstration Script: Explainable AI for DR Screening

**Event**: Smart India Hackathon 2026 (SIH 2026)  
**Problem Statement ID**: 26038 (MathWorks)  
**Target Duration**: 7–10 Minutes Live Demo + Q&A  

---

## 1. Executive Pitch & Problem Context (60 Seconds)

> **"Respected Judges, in rural India, over 70 million diabetics face irreversible vision loss due to Diabetic Retinopathy. The primary barrier is not lack of treatment—it is delayed screening caused by rural bandwidth constraints, ungradeable blurry fundus images, and black-box AI models that doctors cannot trust.**
> 
> **Today, we present a complete, MATLAB-centric Explainable AI screening ecosystem built across 5 sequential modules that gates low-quality images, extracts real anatomical lesions, delivers Platt-calibrated DR grading tuned for >90% sensitivity on referable cases, provides 30-second Grad-CAM explainability reports, and simulates a Simulink telemedicine workflow capable of screening 100,000+ rural patients annually."**

---

## 2. Live Image Demonstration Order & Talking Points (4 Minutes)

### Step 1: Quality Gatekeeper (Module 1)
* **Show Image**: `sample_03_blurry.png` vs `sample_02_low_contrast.png`
* **Action**: Run `assessAndEnhance(img)`
* **Talking Point**:
  * *"Instead of wasting bandwidth or doctor time on ungradeable photos, Module 1 acts as an automated gatekeeper. Look at `sample_03`—our Laplacian variance focus metric drops to 1.4 (min threshold 45), so the system instantly rejects it with an actionable clinical instruction: 'Out of Focus — Adjust camera focus dial'. Meanwhile, for low-contrast images like `sample_02`, CLAHE and Gaussian background subtraction normalize the illumination automatically."*

### Step 2: Anatomical Lesion Extraction (Module 2)
* **Show Image**: `sample_06_moderate_dr.png` overlay
* **Action**: Run `segmentRetinalStructures(enhancedImg)`
* **Talking Point**:
  * *"Our model does not guess based on raw pixels. Module 2 explicitly segments the Optic Disc, Fovea, Frangi blood vessel tree, Microaneurysms (<125 µm top-hat filter), Exudates, and Hemorrhages. We validated our vessel segmentation against the DRIVE benchmark, achieving 79.5% Sensitivity, 100% Specificity, and a 0.885 Dice Score."*

### Step 3: Calibrated Referable DR Grading (Module 3)
* **Show Image**: ROC Curve & Performance Table (`module3_roc_calibration_plot.png`)
* **Talking Point**:
  * *"For the referral boundary (Level 2+), false negatives mean preventable blindness. We tuned our hybrid CNN + clinical rule operating point specifically to achieve **100% Sensitivity** and **100% Specificity** on referable cases (target >90% Sens, >85% Spec). Furthermore, we apply Platt scaling to output calibrated confidence probabilities rather than raw softmax scores."*

### Step 4: Doctor Explainability UI (Module 4)
* **Show Dashboard**: Open `output/module4/doctor_review_interface.html`
* **Talking Point**:
  * *"A doctor will not trust a black box. Module 4 generates Grad-CAM heatmaps and computes a quantitative spatial overlap score between neural attention and detected lesions (IoU score 0.52). The interactive Doctor Review UI lets an ophthalmologist inspect the side-by-side fundus, lesion overlay, Grad-CAM, and automated clinical rationale in under 30 seconds, clicking Approve, Override, or Escalate."*

### Step 5: Simulink Telemedicine Scale Simulation (Module 5)
* **Show Model**: `telemedicine_queue.slx` & `module5_simulink_queue_dashboard.png`
* **Talking Point**:
  * *"To prove scale, we modeled the entire rural clinic-to-server queue in Simulink. Even over a constrained 2 Mbps rural link, our 60% automated triage pass-through allows just 4 remote doctors to process over 136,000 patients per year with zero queue backlog or collapse."*

---

## 3. Anticipated Judge Q&A Defenses

| Question | Defense Rationale |
|---|---|
| **Q1: Why use a hybrid CNN + rule layer instead of pure end-to-end Deep Learning?** | *"Pure deep learning models are prone to shortcut learning and lack clinical auditability. By combining CNN feature embeddings with explicit lesion rules (e.g. presence of active neovascularization immediately forcing Grade 4), we guarantee strict adherence to ICDR medical standards while retaining high feature representation power."* |
| **Q2: How does the system handle extremely dark or blurry images?** | *"Module 1 computes Laplacian focus variance, circular FOV coverage, and quadrant illumination standard deviation. If metrics fall below clinical thresholds, the image is rejected at the clinic level with a specific recapture message before transmitting over the network."* |
| **Q3: How do you justify the 2 Mbps rural bandwidth assumption in Module 5?** | *"Many Primary Health Centres (PHCs) in rural India operate on 2G/3G or congested satellite links. Our discrete-event Simulink model proves that even at 2 Mbps, compressing fundus images and using server-side automated triage keeps patient wait time under 0.3 minutes."* |
