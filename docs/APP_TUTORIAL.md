# System Inspection Litography - User Manual

Welcome to the comprehensive user guide for the System Inspection Litography (SIL) application. This document details every feature, window, and workflow in the system.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
    - [Login](#21-login)
    - [Dashboard](#22-dashboard)
3. [Calibration & Setup](#3-calibration--setup)
    - [Camera Layout](#31-camera-layout)
    - [Step 1: Camera Setup](#32-step-1-camera-setup)
    - [Step 2: Corner Detection](#33-step-2-corner-detection)
    - [Step 3: Rectification](#34-step-3-rectification)
    - [Step 4: Crop Tuning](#35-step-4-crop-tuning)
    - [Step 5: Resize](#36-step-5-resize)
    - [Step 6: Alignment (Critical)](#37-step-6-alignment-critical)
4. [Dataset Management](#4-dataset-management)
    - [Managing SKUs](#41-managing-skus)
    - [Capturing Images](#42-capturing-images)
    - [Validation Logic](#43-validation-logic)
    - [Exports & Reports](#44-exports--reports)
5. [Inspection Mode](#5-inspection-mode)
    - [Starting an Inspection](#51-starting-an-inspection)
    - [Live Monitoring](#52-live-monitoring)
    - [History & Logs](#53-history--logs)
    - [Defects Gallery](#54-defects-gallery)
6. [User Management](#6-user-management)

---

## 1. Introduction
SIL is an advanced computer vision system designed for real-time quality inspection of lithography on cans. It uses AI (PatchCore) to detect subtle defects like scratches, dents, and printing errors.

---

## 2. Getting Started

### 2.1 Login
Upon launching the application (`./run_app.sh`), you are greeted by the Login Screen.
- **Username/Password**: Enter your credentials.
- **Roles**:
    - **Operator**: Access only to Inspection Mode.
    - **Tecnico**: Access to Inspection and Calibration.
    - **Admin/Master**: Full access, including User Management.

### 2.2 Dashboard
The Dashboard is your central hub. It displays:
- **Current User**: Shown in the top right.
- **Action Cards**:
    - **🔍 Inspection**: Run production jobs.
    - **📸 Dataset**: Collect training data.
    - **⚙️ Calibration**: Configure camera and AI pipeline (Admin/Tecnico only).
    - **👥 Users**: Manage accounts (Admin only).

> **Note**: If an inspection is running in the background, you will be prompted to stop it before entering Calibration or Dataset modes to free up the camera.

---

## 3. Calibration & Setup
**Window**: `AdjustmentWindow`
**Access**: Admin / Scientific / Tecnico

This is the most critical part of the system. If calibration is poor, inspection will fail. The wizard guides you through 6 steps.

### 3.1 Camera Layout
- **Left Panel**: Live Video Feed or Frozen Capture.
- **Right Panel**: Controls for the current step.

### 3.2 Step 1: Camera Setup
Adjust the physical camera properties.
- **Focus Mode**: Check `🔍 Focus Mode` to switch to high-resolution for precise manual focusing.
- **Sharpness Bar**: Use the visual bar to maximize the focus score.
- **Sliders**:
    - **Exposure**: Brightness of the image. Too high = reflections; Too low = noise.
    - **Brightness/Contrast**: Fine-tune contrast for better feature detection.
- **Freeze**: Click `Show Capture (Freeze)` to take a snapshot for the next steps.

### 3.3 Step 2: Corner Detection
Detects the 4 corners of the metal sheet.
- **Test Detection**: Draws **Green Crosshairs** on the 4 corners.
    - *Tip*: If the crosshairs aren't on the corners, adjust the `ROI Size` or `Margins`.
- **Params**:
    - `Margins`: How far from the image edge to look.
    - `skew`: Adjusts the search box position for angled cameras.

### 3.4 Step 3: Rectification
Unglosses the image to make the sheet look flat (top-down view).
- **Sheet Skew**: Adjust this if vertical lines on the can look tilted in the preview.

### 3.5 Step 4: Crop Tuning
Defines where the cans are cut out from the sheet.
- **Green Grid**: Shows exactly where the system will crop.
- **Params**:
    - `First Can X/Y`: Position of the top-left can.
    - `Step X/Y`: Distance between cans.
    - **Goal**: Center the can **perfectly** inside the green box.

### 3.6 Step 5: Resize
Standardizes image size (default 448x448) for the AI model.

### 3.7 Step 6: Alignment (Critical)
The AI needs the can to be perfectly aligned to compare it with a "Gold Standard".
- **Test Alignment**: Matches the current can against a Reference Image.
- **📸 Capture As New Reference**:
    - **Use this feature!** After optimizing camera/focus in Step 1, come here and click this button.
    - It saves currently viewed can as the **Master Reference**.
    - This guarantees **100% Alignment Score** for your specific setup.

---

## 4. Dataset Management
**Window**: `DatasetWindow`

Used to collect "Good" images to train the AI.

### 4.1 Managing SKUs
- **Dropdown**: Select the product you are working on (e.g., "Bom Petisco Azeite").
- **➕ New SKU**: Create a new product profile. Requires a unique name.

### 4.2 Capturing Images
1.  Place a **GOOD** can on the fixture.
2.  Click **Capture & Process**.
3.  The system runs the full pipeline (Crop -> Align -> Save).

### 4.3 Validation Logic
The system automatically checks image quality before saving:
- **Blur**: Rejects out-of-focus images.
- **Reflection**: Checks for "specular highlights" (pure white pixels).
    - *Note*: Thresholds are relaxed (254 brightness, <5000 pixels) to allow metallic shine.
- **Alignment**: Ensures the can matches the Reference Image.
- **Outcome**:
    - **Good** -> Saved to `dataset/train/`.
    - **Bad/Rejected** -> Saved to `dataset/debug/` (useful for diagnosing why it failed).

### 4.4 Exports & Reports
- **Export Dataset**: Zips the folder for transfer to the training PC.
- **Generate Report**: Creates a PDF summary of the dataset statistics.

---

## 5. Inspection Mode
**Window**: `InspectionWindow`

The production interface.

### 5.1 Starting an Inspection
1.  **Select SKU**: Choose the product to inspect.
2.  **Order Info**: Click **Nova OP** to enter Production Order details (Operator Name, Lot ID).
3.  **Start**: Click **▶ INICIAR**. The system begins processing camera frames.

### 5.2 Live Monitoring
- **Main View**: Shows the live feed with overlay boxes.
    - 🟩 **OK**: Good can.
    - 🟥 **NOK**: Defective can.
- **Heatmap**: Shows *where* the defect is (red glowing spots).
- **Detail View**: Click any result in the "History" list to see the full analysis, including anomaly score and threshold.

### 5.3 History & Logs
- **Right Panel**: Lists the last ~50 inspected items.
- **Logs Button**: Opens a full textual log of events and errors.

### 5.4 Defects Gallery
Click **Galeria de Defeitos** to browse past rejections.
- Organized by **Year / Month**.
- Helps verify false positives/negatives.

---

## 6. User Management
**Window**: `UserManagementDialog`
**Access**: Admin / Master

Manage system access.
- **Add User**: Create new logins.
- **Roles**:
    - `operator`: Can only run inspections.
    - `tecnico`: Can adjust camera/dataset.
    - `admin`: Can change system configs.
    - `master`: Can delete other admins.

---

*Generated by Antigravity - 2026*
