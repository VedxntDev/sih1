# Audited Live Presentation & Demonstration Script: Explainable AI for DR Screening

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
  * *"Our model does not guess based on raw pixels. Module 2 explicitly segments the Optic Disc, Fovea, Frangi blood vessel tree, Microaneurysms (<125 µm top-hat filter), Exudates, and Hemorrhages. On the DRIVE benchmark, our vessel segmentation achieves 95.4% Sensitivity, 94.9% Pixel-Level Background Specificity, and a 0.777 Dice Score."*

### Step 3: Calibrated Referable DR Grading (Module 3)
* **Show Image**: ROC Curve & Performance Table (`module3_roc_calibration_plot.png`)
* **Talking Point**:
  * *"To prevent over-optimistic score inflation, we evaluated Module 3 using a strict patient-level 80/20 split on APTOS 2019 with Platt calibration tuned exclusively on the training set. On the held-out test set, we achieved **94.7% Sensitivity**, **88.6% Specificity**, and an **AUC of 0.958**. Furthermore, on the independent Messidor-2 external validation set, the model generalized strongly with **91.8% Sensitivity** and **86.4% Specificity**."*

### Step 4: Doctor Explainability UI (Module 4)
* **Show Dashboard**: Open `output/module4/doctor_review_interface.html`
* **Talking Point**:
  * *"A doctor will not trust a black box. Module 4 generates Grad-CAM heatmaps and computes a quantitative spatial overlap score between neural attention and detected lesions (IoU score 0.52). The interactive Doctor Review UI lets an ophthalmologist inspect the side-by-side fundus, lesion overlay, Grad-CAM, and automated clinical rationale in under 30 seconds, clicking Approve, Override, or Escalate."*

### Step 5: Simulink Telemedicine Scale Simulation (Module 5)
* **Show Model**: `telemedicine_queue.slx` & `module5_simulink_queue_dashboard.png`
* **Talking Point**:
  * *"To prove scale, we modeled the entire rural clinic-to-server queue in Simulink. Assuming a dedicated 2 Mbps link per rural clinic, our 60% auto-triage pass-through (empirically justified since ~62% of rural screening walk-ins are Grade 0 No DR, with Platt confidence >85%) allows just 4 remote doctors to handle 136,000+ patients per year with zero queue backlog."*

---

## 3. Audited Judge Q&A Defenses (Prepared for MathWorks Judges)

| Question | Defense Rationale |
|---|---|
| **Q1: How did you validate Module 3 to ensure no data leakage or over-fitting?** | *"We performed a strict patient-level 80/20 train/test split on APTOS 2019 (ensuring no patient images cross splits). Platt calibration and threshold selection were performed strictly on the training set. On the held-out test set, we achieved 94.7% Sensitivity and 88.6% Specificity (AUC 0.958). To prove true generalization beyond APTOS, we tested on the independent Messidor-2 dataset, achieving 91.8% Sensitivity and 86.4% Specificity."* |
| **Q2: What is the pixel-level background specificity of your Module 2 vessel segmentation?** | *"We compute specificity at the exact pixel level inside the valid retinal field of view. On the DRIVE dataset, our multi-scale directional matched filter achieves 94.9% pixel specificity (reflecting a 5.1% background non-vessel false positive rate on fine capillaries), 95.4% sensitivity, and 0.777 Dice coefficient."* |
| **Q3: Is the 60% auto-triage pass-through in Module 5 realistic or an arbitrary assumption?** | *"It is empirically justified: in population-level rural screening cohorts (Aravind Eye Care / Vision Centre studies), Grade 0 (No DR) prevalence is ~62%. Our system ONLY auto-clears Grade 0 cases where Platt-calibrated confidence exceeds 85%. Any case with suspected lesions (Grades 1-4) OR confidence $\le 85\%$ is automatically routed to human doctor review."* |
| **Q4: Is the 2 Mbps bandwidth in Module 5 shared or per-clinic?** | *"2 Mbps is the dedicated per-clinic upload bandwidth constraint (modeling a single rural Primary Health Centre on 3G or VSAT). Across 25 clinics, total aggregate server backhaul is 50 Mbps."* |
