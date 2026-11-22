# Coursework Task Tracking: FCVG101 Computer Vision

## Project Overview
- **Scenario**: Visual Inspection Assistant (Facial Expression & Camera Geometry Inspection Pipeline)
- **Dataset**: 
  - Checkerboard Calibration Dataset: OpenCV Standard Calibration Set (`calibration_images/`, 13 images, $9\times6$ inner corners, `square_size = 1.0` grid unit)
  - Inspection & Recognition Dataset: FER2013 (`archive/`)
- **Key Deliverables**: 
  - Task 1: Classical CV Pipeline (Calibration, Filtering, Gradients, Frequency, Feature Detection & Matching) — 20 points
  - Task 2: Deep Learning Model & Data Pipeline — 35 points
  - Task 3: Assistant Interface (Streamlit/Gradio) — 30 points
  - Documentation & Submission — 15 points

---

## Section 2: Classical Computer Vision Pipeline (Task 1: Core Image Analysis Engine)

- [x] **2.0 Image Formation and Camera Calibration** *(Coursework Rubric Criterion 1.1, Requirement a - 5 points)*
  - [x] **2.0.1 Homogeneous Pinhole Model & Distortion Formulation**: Formulated $s \tilde{\mathbf{x}} = K [R \mid \mathbf{t}] \tilde{\mathbf{X}}_w$ with scale factor $s=Z_c$; distinguished linear pinhole projection from non-linear normalized distortion ($k_1, k_2, k_3, p_1, p_2$).
  - [x] **2.0.2 Checkerboard Ingestion & Corner Detection**: Ingested 13 views from OpenCV standard sequence (`left01.jpg`–`left14.jpg`). Verified $9\times6$ inner corners (54 corners per view), filtered strictly with `ret == True` achieving **13/13 (100%)** successful sub-pixel detections.
  - [x] **2.0.3 Camera Calibration Execution (`cv2.calibrateCamera`)**: 
    - **Overall RMS Reprojection Error**: `0.4087 pixels` (< 0.5 px benchmark)
    - **Intrinsic Matrix $K$**:
      $$\begin{bmatrix} 536.07 & 0 & 342.37 \\ 0 & 536.02 & 235.54 \\ 0 & 0 & 1 \end{bmatrix}$$
    - **Focal Lengths**: $f_x = 536.07\text{ px}, f_y = 536.02\text{ px}$ (aspect ratio $f_x/f_y \approx 1.0001$)
    - **Principal Point**: $c_x = 342.37\text{ px}, c_y = 235.54\text{ px}$ (near image center $320, 240$)
    - **Distortion Vector $D$**: $k_1 = -0.2651, k_2 = -0.0467, p_1 = 0.0018, p_2 = -0.0003, k_3 = 0.2523$
  - [x] **2.0.4 Image Undistortion & Collinearity Proof**: Computed `new_K` with $\alpha = 0$; demonstrated **94.4% reduction in corner curvature deviation** (from 2.848 px in distorted view down to 0.158 px in undistorted view) along outer boundary chords.
  - [x] **2.0.5 Quantitative Evaluation**: Plotted **Reprojection RMSE per View** bar chart against the global Overall RMS baseline (`0.4087 px`). Min view RMSE: `0.2312 px`, Max view RMSE: `0.5891 px`.

- [x] **2.1 Preparation and Image Loading**
  - Read input image, verify dimensions, channels, data types, pixel min/max.

- [x] **2.2 Grayscale Conversion & Inspection (2.1.2)**
  - Transform BGR to Grayscale, inspect luminance distribution.

- [x] **2.3 Image Filtering & Denoising** *(Coursework Rubric Criterion 1.2 - 5 points)*
  - Add Gaussian noise ($\sigma=20$).
  - Evaluate Gaussian filter ($3\times3, 9\times9$) and Median filter ($3\times3$) with MSE metrics.

- [x] **2.4 Image Gradients & Edge Detection** *(Coursework Rubric Criterion 1.2)*
  - Gaussian smoothing + Sobel gradients ($G_x, G_y$, Magnitude) + Canny edge detection.

- [x] **2.5 Frequency Analysis & Filtering** *(Coursework Rubric Criterion 1.2)*
  - 2D FFT (`np.fft.fft2`), spectrum shifting, Low-pass & High-pass mask filtering, image reconstruction (MSE evaluation).

- [x] **2.6 Feature Detection and Matching** *(Coursework Rubric Criterion 1.3, Requirement b - 5 points)*
  - [x] **2.6.1 Image Pair Selection**: Loaded sample from `archive/train/happy/Training_10019449.jpg`, scaled to $256\times256$, and generated multi-view inspection View 2 ($15^\circ$ rotation, $1.05\times$ scale).
  - [x] **2.6.2 Selected Detector and Descriptor**: Implemented both SIFT (128-D float vector) and ORB (256-bit binary string) with scale/orientation rich keypoint visualizations.
  - [x] **2.6.3 Feature Matching and Ratio Test**: Implemented BFMatcher (L2 distance for SIFT, Hamming distance for ORB) with $k$-NN ($k=2$) and Lowe's Ratio Test ($\tau = 0.75$).
  - [x] **2.6.4 Matching Results and Quantitative Analysis**: Visualized match correspondences and produced benchmark comparison table:
    - **SIFT**: 53 keypoints (View 1), 54 keypoints (View 2), 36 matches retained (**67.92% retention rate**).
    - **ORB**: 99 keypoints (View 1), 87 keypoints (View 2), 51 matches retained (**51.52% retention rate**), matching speed >2x faster than SIFT.

---

## Section 3: Deep-Learning Model and Data Pipeline (Task 2 — 35 Points)

- [x] **3.1 Classification Task and Dataset Selection** *(Criteria 2.1 & 2.2)*
  - [x] **3.1.1 Target Classes and Expected Outputs**: 7 affective classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) with probability vector output via Softmax.
  - [x] **3.1.2 Dataset Source and Rationale**: FER2013 chosen for real-world facial inspection diversity under varying poses, lighting, and occlusions.
  - [x] **3.1.3 Dataset Statistics & Class Distribution**: Exact splits identified:
    - Train: 28,709 samples (strictly filtering out 2 non-jpg screenshots).
    - Validation (`PublicTest`): 3,589 samples.
    - Test (`PrivateTest`): 3,589 samples.
    - Extreme class imbalance quantified: `disgust` (436 samples, 1.52%) vs. `happy` (7,215 samples, 25.13%).

- [x] **3.2 Data Pipeline & Preprocessing** *(Criterion 2.1)*
  - [x] **3.2.1 Data Ingestion & Sanitization**: Regex filtering `^Training_\d+\.jpg$` ensures 100% clean data ingestion without cross-split leakage.
  - [x] **3.2.2 Train/Val/Test Isolation**: Pre-partitioned benchmarks preserved; no random reshuffling of original splits.
  - [x] **3.2.3 Normalization**: Computed strictly on Train split ($\mu_{\text{train}} = 0.507739, \sigma_{\text{train}} = 0.255078$), saved to `train_stats.json`, and strictly applied to Val, Test, and Task 3 inference.
  - [x] **3.2.4 Data Augmentation**: Fast batch-level augmentation (random horizontal flip $p=0.5$ + translation $\pm 3\text{ px}$) applied exclusively to training batches.

- [x] **3.3 Model Design & Architecture** *(Criterion 2.2)*
  - [x] **3.3.1 Architecture**: `EmotionCNN` featuring 4 convolutional stages ($1 \to 32 \to 64 \to 128 \to 256$), each with Conv2d ($3\times3$), BatchNorm2d, ReLU, and MaxPool2d ($2\times2$).
  - [x] **3.3.2 Regularization**: Dropout (0.25 on blocks 3-4, 0.5 on dense layer) and L2 weight decay ($10^{-4}$) to prevent overfitting. Total trainable parameters: **617,063**.

- [x] **3.4 Model Training & Checkpointing** *(Criterion 2.3)*
  - [x] **3.4.1 Loss Function**: Cost-sensitive Cross-Entropy with inverse frequency weights (weight for `disgust` = 4.0, `happy` = 0.5684) to counteract extreme class imbalance.
  - [x] **3.4.2 Optimization**: AdamW optimizer ($\text{lr} = 10^{-3}$) with `ReduceLROnPlateau` scheduler.
  - [x] **3.4.3 Checkpoint Selection**: Monitored on validation loss. Best checkpoint selected at Epoch 7 with `Val Loss: 1.2899`, `Val Acc: 50.46%`.

- [x] **3.5 Model Evaluation & Error Analysis** *(Criteria 2.3 & 2.4)*
  - [x] **3.5.1 Training/Validation Curves**: Plotted and saved to `model/learning_curves.png`.
  - [x] **3.5.2 Performance Metrics on PrivateTest (Unseen)**:
    - **Overall Test Accuracy**: **51.60%**
    - **Macro-averaged F1 Score**: **0.4402**
    - **Weighted-average F1 Score**: **0.4907**
    - **Disgust Recall**: Lifted to **38.18%** (preventing minority class collapse).
  - [x] **3.5.3 Confusion Matrix**: Generated raw and normalized matrices saved to `model/confusion_matrix.png`.
  - [x] **3.5.4 Checkpoint & Pipeline Integrity**: Exported standalone `model/fer2013_checkpoint.pth` containing model weights, 7 class names, and normalization stats; reloaded and validated with independent test inference.

---

## Section 4: Visual Inspection Assistant Application (Task 3 — 30 Points)

- [x] **4.1 Architecture & Workflow**: Modular design with reusable engines in `modules/` (`calibration_engine.py`, `filter_engine.py`, `feature_engine.py`, `emotion_classifier.py`).
- [x] **4.2 Interface Design & Input Validation**:
  - [x] **4.2.1 Image and Camera Input Validation**: Triple input support (Upload, Browser Webcam Snapshot via `st.camera_input`, and Dataset Samples). Strict validation rejecting 0-byte and corrupted non-image files with graceful `st.error`.
  - [x] **4.2.2 Face Detection and Cropping**: OpenCV 5.0.0 YuNet DNN detector (`cv2.FaceDetectorYN`) with aspect-ratio preserving scaling bounded at $\text{max\_dim} = 640\text{ px}$. Bounding boxes rescaled and clamped via `np.clip` to original coordinates. Configurable heuristic margin ($0\% - 25\%$, default $10\%$).
  - [x] **4.2.3 Task Selection and Parameter Controls**: Explicit Mode Selector (`Ảnh thường cần phát hiện mặt` vs `Ảnh chân dung đã crop sẵn (Bypass)`). Interactive sliders for Score Threshold and Heuristic Margin. Multi-face selector dropdown for choosing individual faces or analyzing all.
  - [x] **4.2.4 Result Visualization**: Dual-view comparison (Full scene with indexed bounding boxes, cropped ROI preview, normalized $48 \times 48$ tensor view, and latency readout).
- [x] **4.3 Integration of the Task 1 Pipeline**:
  - [x] **4.3.1 Camera Calibration and Undistortion**: Interactive checkerboard calibration, intrinsic matrix $K$, distortion $D$, RMSE readout ($0.4087\text{ px}$), and live image undistortion.
  - [x] **4.3.2 Filtering, Gradients and Edge Detection**: Adjustable Gaussian kernel size, Median filter, Sobel $G_x/G_y$ gradients, and interactive Canny thresholds ($T_1, T_2$).
  - [x] **4.3.3 Frequency Analysis**: 2D FFT magnitude spectrum, interactive cutoff radius $D_0$, and Ideal vs. Gaussian Low-pass / High-pass reconstructed image views.
  - [x] **4.3.4 Feature Detection and Matching**: SIFT vs. ORB detector selection, $k$-NN ratio test threshold $\tau$, side-by-side keypoint display, and correspondence matching visualization.
- [x] **4.4 Integration of the Task 2 Model**:
  - [x] **4.4.1 Model Loading**: Cached `EmotionClassifier` loader checking `model/fer2013_checkpoint.pth` with robust fallbacks.
  - [x] **4.4.2 Face Cropping and Model Preprocessing**: Cropped ROI converted to Grayscale, resized via `cv2.INTER_AREA` to $48 \times 48$, and standardized strictly with training set parameters $(\mu=0.507739, \sigma=0.255078)$ into `(1, 1, 48, 48)` float32 tensor.
  - [x] **4.4.3 Predicted Emotion and Probability Display**: Top-1 predicted emotion class, horizontal Softmax probability distribution bar chart, **Heuristic Low-Score Warning** when Top-1 score $< 40\%$, and explicit Academic Disclaimer on scope.
- [x] **4.5 Error Handling and Input Validation**: Graceful handling for non-face images (`st.warning`), corrupted inputs (`st.error`), and multi-face index safety with 0 unhandled exceptions.
- [x] **4.6 Application Testing and Demonstration**:
  - [x] **4.6.1 Webcam and Face Detection Test Cases**: Automated verification test suite (`scratch/test_task3_cases.py` & in-app GUI tab) covering TC1 to TC6 with **100% PASS** on measurable criteria.
  - [x] **Verified Resolution**: Đã xác nhận hoạt động đúng trên ảnh kiểm thử camera $640 \times 480$.
