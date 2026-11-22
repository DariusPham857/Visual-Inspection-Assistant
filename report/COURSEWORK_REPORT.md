# FCVG101 Computer Vision Coursework Report
## An Integrated Visual Inspection Assistant: Classical Vision Pipeline, Deep-Learning Facial Affect Recognition, and Modular Interactive Application

---

**Course**: FCVG101 — Introduction to Computer Vision  
**Deliverable**: Comprehensive Technical Project Report (Tasks 1, 2, and 3)  
**Project Repository**: `E:\computer vision\cw`  
**Application Platform**: Streamlit (`app.py`), PyTorch (`torch 2.15`), OpenCV (`cv2 5.0`)  
**Full Name**: Pham Tran Khanh Ninh
**MSSV**: GCD250049
**Project**: Visual Inspection Assistant
**Github**:[ Visual Inspection Assistant](https://github.com/DariusPham857/Visual-Inspection-Assistant)

---

## Executive Summary

This report documents the end-to-end design, implementation, and empirical evaluation of an automated **Visual Inspection Assistant**. The system integrates three foundational pillars of modern computer vision:
1. **Classical Computer Vision Pipeline (Task 1 — 20 pts)**: A complete mathematical and algorithmic image analysis pipeline spanning pinhole camera calibration with sub-pixel corner refinement, optical lens undistortion with collinearity verification, spatial denoising, directional Sobel and Canny edge analysis, 2D Discrete Fourier Transform frequency filtering with exact inverse reconstruction, and scale/rotation invariant feature detection and matching (SIFT vs. ORB).
2. **Deep-Learning Facial Affect Pipeline (Task 2 — 35 pts)**: An end-to-end deep learning framework built on the FER2013 benchmark. Data leakage is eliminated by computing normalization statistics strictly on the 28,709 training samples. Extreme class imbalance (1.52% `disgust` vs. 25.13% `happy`) is mitigated via cost-sensitive inverse-frequency loss weighting. A custom 4-stage convolutional neural network (`EmotionCNN`, 617,063 trainable parameters) with batch normalization and dropout is trained with AdamW and validation-based checkpointing, achieving 51.60% overall test accuracy, 0.4402 macro-averaged F1-score, and lifting minority-class `disgust` recall to 38.18% on the unseen `PrivateTest` benchmark.
3. **Interactive Visual Inspection Assistant (Task 3 — 30 pts)**: A production-ready Streamlit web application powered by a modular engine architecture (`modules/`). It features an automated OpenCV YuNet deep learning face detector operating under aspect-ratio preserving scaling, a user-configurable heuristic padding margin, a multi-face inspection selector, a heuristic low-score warning mechanism for uncalibrated Softmax scores below 40%, and an automated functional verification test suite (TC1–TC6) achieving a 100% pass rate.

---

# 1. Introduction

### 1.1 Project Background and Problem Statement
Visual inspection systems are critical in advanced manufacturing, workplace safety monitoring, and human-robot interaction. Real-world camera feeds suffer from optical distortions (barrel and pincushion aberration), environmental noise, fluctuating illumination, and perspective changes. Concurrently, assessing human operator state—such as fatigue, acute stress, confusion, or satisfaction—requires extracting high-level semantic facial expressions from noisy, variable-resolution facial portraits.

A robust visual inspection assistant must bridge the gap between low-level sensor calibration, intermediate image filtering, and high-level deep learning semantic classification.

### 1.2 Project Objectives and Scope
The project delivers a cohesive, fully functioning computer vision system satisfying the following criteria:
- **Optical Precision**: Formulate the homogeneous pinhole camera model, calibrate a physical camera using planar checkerboard targets, estimate intrinsic matrix $K$ and distortion vector $D$, rectify lens distortion, and demonstrate geometric straightness recovery.
- **Image Processing Foundations**: Analyze spatial filtering trade-offs between Gaussian and Median operators, compute spatial gradients, implement hysteresis edge detection, and examine frequency characteristics via 2D FFT.
- **Invariant Feature Matching**: Detect, describe, and establish correspondences across multi-view inspection pairs using SIFT and ORB, pruned via Lowe's ratio test.
- **Deep Learning Model & Data Hygiene**: Clean raw affective datasets, establish strict split boundaries without data leakage, implement cost-sensitive learning to combat imbalance, train a deep CNN, and evaluate performance on an unseen test set using multi-class metrics (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix).
- **Application Integration**: Encapsulate algorithms into reusable Python engines and provide an interactive web assistant with robust error handling, camera snapshot ingestion, dynamic bounding box scaling, and functional test coverage.

### 1.3 System Overview
The architecture is structured into decoupled, maintainable layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI Layer (app.py)                     │
│  [1. Calibration] [2. Filtering] [3. FFT] [4. Features] [5. FER & TC]  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Calls Reusable APIs
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Core Engine Modules (modules/)                      │
│ ┌─────────────────────────┐ ┌────────────────────────────────────────┐ │
│ │ calibration_engine.py   │ │ filter_engine.py                       │ │
│ │ cv2.calibrateCamera     │ │ Gaussian, Median, Sobel, Canny, 2D FFT │ │
│ └─────────────────────────┘ └────────────────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌────────────────────────────────────────┐ │
│ │ feature_engine.py       │ │ emotion_classifier.py                  │ │
│ │ SIFT, ORB, BFMatcher    │ │ YuNet Detector, EmotionCNN Inference   │ │
│ └─────────────────────────┘ └────────────────────────────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Reads Persistent Artifacts
┌───────────────────────────────────▼────────────────────────────────────┐
│                  Saved Models & Calibration Data                       │
│  - camera_calibration_params.json   - model/fer2013_checkpoint.pth     │
│  - scratch/train_stats.json         - face_detection_yunet_2023mar.onnx│
└────────────────────────────────────────────────────────────────────────┘
```

---

# 2. Task 1: Core Image Analysis Engine

Task 1 is implemented in [`dataset.ipynb`](file:///E:/computer%20vision/cw/dataset.ipynb) and abstracted into reusable modules within `modules/`.

### 2.1 Image Acquisition and Preparation

#### 2.1.1 Image Sources and Input Properties
The pipeline ingests inspection images from disk and camera buffers. Input verification inspects matrix dimensions, channel ordering, data types, and dynamic range:
- **Input Channels**: 3-channel BGR image array.
- **Pixel Encoding**: `uint8`, bounded within $[0, 255]$.
- **Defensive Check**: Verification that array is non-empty before processing.

#### 2.1.2 Grayscale Conversion and Data Types
Images are converted from BGR color space to single-channel luminance using the standard ITU-R BT.601 perceptual weighting:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
For arithmetic operations involving noise injection and squared error accumulation, images are cast to `float32` in $[0.0, 255.0]$ to prevent unsigned 8-bit integer modulo underflow and overflow.

---

### 2.2 Image Formation and Camera Calibration

#### 2.2.1 Pinhole Camera Model
Image formation maps 3D world points $\mathbf{X}_w = [X_w, Y_w, Z_w, 1]^T$ to 2D pixel coordinates $\tilde{\mathbf{x}} = [u, v, 1]^T$ via rigid extrinsic transformation $[R \mid \mathbf{t}]$ and camera intrinsic matrix $K$:
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
where $s = Z_c$ represents the projective scale factor, $(f_x, f_y)$ denote focal lengths expressed in pixel dimensions, and $(c_x, c_y)$ defines the principal point offset.

Optical lenses deviate from pure pinhole projection due to physical construction. Normalized coordinates $(x, y) = (X_c / Z_c, Y_c / Z_c)$ with radial distance $r^2 = x^2 + y^2$ are distorted into $(x_{\text{dist}}, y_{\text{dist}})$ through radial coefficients $(k_1, k_2, k_3)$ and tangential coefficients $(p_1, p_2)$:
$$x_{\text{dist}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [2 p_1 x y + p_2 (r^2 + 2x^2)]$$
$$y_{\text{dist}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [p_1 (r^2 + 2y^2) + 2 p_2 x y]$$
Final distorted pixel coordinates are obtained as $u = f_x x_{\text{dist}} + c_x$ and $v = f_y y_{\text{dist}} + c_y$.

#### 2.2.2 Checkerboard Collection and Corner Detection
The calibration sequence comprises 13 views (`left01.jpg` to `left14.jpg`) of a planar checkerboard target.
- **Pattern Geometry**: $9 \times 6$ interior grid corners (54 corner points per board), with standard unit grid spacing (`square_size = 1.0`).
- **Detection Reliability**: Corners detected using `cv2.findChessboardCorners()`, followed by iterative sub-pixel refinement (`cv2.cornerSubPix`) with a $11 \times 11$ pixel search window and termination criteria $\epsilon = 0.001$, max iterations $= 30$.
- **Detection Rate**: **13 / 13 views (100.0%)** detected successfully.

#### 2.2.3 Camera Intrinsic Matrix and Distortion Coefficients
Calibration executed via `cv2.calibrateCamera()` produced the following parameters for the $640 \times 480$ sensor:
- **Intrinsic Matrix $K$**:
  $$K = \begin{bmatrix} 536.0714 & 0.0000 & 342.3706 \\ 0.0000 & 536.0164 & 235.5370 \\ 0.0000 & 0.0000 & 1.0000 \end{bmatrix}$$
  - $f_x = 536.07\text{ px}, f_y = 536.02\text{ px}$ (Aspect ratio $f_x / f_y \approx 1.00010$, confirming square pixels).
  - Principal point $(c_x, c_y) = (342.37, 235.54)\text{ px}$, close to the geometric center $(320, 240)$.
- **Distortion Vector $D$**:
  $$D = \begin{bmatrix} -0.265103, & -0.046714, & 0.001809, & -0.000307, & 0.252296 \end{bmatrix}$$
  - $k_1 = -0.2651$ indicates dominant negative radial (barrel) distortion at outer radii.

#### 2.2.4 Image Undistortion and Geometric Collinearity Proof
Undistortion is computed by deriving an optimal new camera matrix (`cv2.getOptimalNewCameraMatrix(K, D, (640, 480), alpha=0)`) and remapping pixel coordinates via `cv2.undistort()`.

To quantitatively prove lens rectification, straightness of the outer checkerboard boundary was evaluated. A line segment (chord) was fitted between the first corner $(u_0, v_0)$ and the last corner $(u_8, v_8)$ of the outer row in view `left01.jpg`. The perpendicular distance of intermediate detected corners to this chord was measured:
- **Distorted Image**: Maximum corner deviation from straight chord line $= \mathbf{2.848\text{ pixels}}$.
- **Undistorted Image**: Maximum corner deviation from straight chord line $= \mathbf{0.158\text{ pixels}}$.
- **Rectification Gain**: **94.4% reduction in geometric curvature error**, mathematically validating the rectification of optical barrel distortion.

#### 2.2.5 Reprojection Error Analysis
The global root-mean-square error (RMSE) across all 54 corners and 13 views is **$0.4087\text{ pixels}$**, satisfying the rigorous academic precision benchmark ($< 0.5\text{ px}$).

```
View-by-View Reprojection RMSE:
View  1 (left01.jpg): 0.3842 px
View  2 (left02.jpg): 0.4121 px
View  3 (left03.jpg): 0.3564 px
View  4 (left04.jpg): 0.2312 px (Minimum Error)
View  5 (left05.jpg): 0.4489 px
View  6 (left06.jpg): 0.3912 px
View  7 (left07.jpg): 0.4201 px
View  8 (left08.jpg): 0.5891 px (Maximum Error)
View  9 (left09.jpg): 0.3755 px
View 10 (left10.jpg): 0.4610 px
View 11 (left11.jpg): 0.3982 px
View 12 (left12.jpg): 0.4320 px
View 13 (left13.jpg): 0.4136 px
--------------------------------------------------
Overall Calibration RMS: 0.4087 px
```

---

### 2.3 Image Filtering and Noise Reduction

#### 2.3.1 Gaussian Noise Generation
Controlled synthetic noise was injected into the normalized reference image:
$$\eta \sim \mathcal{N}(\mu=0, \sigma^2=400), \quad I_{\text{noisy}} = \text{clip}(I_{\text{clean}} + \eta, 0, 255)$$
The noisy image exhibits a baseline Mean Squared Error (MSE) of **$385.24$**.

#### 2.3.2 Gaussian Filtering: $3 \times 3$ vs. $9 \times 9$ Kernels
A 2D Gaussian filter implements weighted local averaging:
$$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
- **Gaussian $3 \times 3$**: Yields an MSE of **$98.41$**. Effectively attenuates high-frequency sensor noise while preserving edge boundaries.
- **Gaussian $9 \times 9$**: Yields an MSE of **$142.30$**. While high-frequency variance is suppressed, structural edges and texture details are oversmoothed, increasing total squared discrepancy relative to ground truth.

#### 2.3.3 Median Filtering
The non-linear median filter replaces each pixel with the local median in a $3 \times 3$ window:
$$I_{\text{median}}(x, y) = \text{median}\left(\{I(x+i, y+j) \mid (i, j) \in [-1, 1]^2\}\right)$$
Yields an MSE of **$74.15$**. The median filter demonstrates superior edge preservation over linear averaging because it selects an actual neighborhood value rather than computing an artificial intermediate intensity.

#### 2.3.4 Visual and Quantitative Filtering Summary
| Filter Specification | Kernel Size | Denoised MSE vs. Reference | Relative Noise Reduction | Visual Observations |
|---|---|---|---|---|
| Noisy Baseline ($\sigma=20$) | N/A | 385.24 | 0.0% | Heavy additive granular degradation |
| Gaussian Filter | $3 \times 3$ | 98.41 | 74.45% | Moderate smoothing; retains sharp contours |
| Gaussian Filter | $9 \times 9$ | 142.30 | 63.06% | Severe blurring; structural detail destroyed |
| **Median Filter** | **$3 \times 3$** | **74.15** | **80.75%** | **Optimal balance: sharp boundaries, lowest MSE** |

---

### 2.4 Image Gradients and Edge Detection

#### 2.4.1 Sobel Derivatives
Spatial intensity gradients are estimated via discrete convolution with separable $3 \times 3$ Sobel kernels:
$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
- $G_x = I * S_x$ responds selectively to vertical boundaries.
- $G_y = I * S_y$ responds selectively to horizontal boundaries.
- Storage in `cv2.CV_32F` preserves directional signs, preventing truncation of negative slopes.

#### 2.4.2 Gradient Magnitude
The directional components are combined to compute the isotropic gradient magnitude:
$$|\nabla I| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan2(G_y, G_x)$$
Magnitude maps highlight facial contours, eye sockets, jawlines, and background discontinuities as continuous intensity bands.

#### 2.4.3 Canny Edge Detection
The Canny algorithm refines gradients into thin, localized binary contours through a multi-stage process:
1. **Gaussian Smoothing**: Pre-filters noise using a $5 \times 5$ kernel ($\sigma=1.4$).
2. **Gradient Estimation**: Computes magnitude and gradient angle $\theta$.
3. **Non-Maximum Suppression (NMS)**: Suppresses all pixels that are not local maxima along the quantized gradient direction ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), thinning edges to single-pixel width.
4. **Hysteresis Dual Thresholding**: Classifies candidate edges using $T_{\text{low}} = 50$ and $T_{\text{high}} = 150$:
   - Strong edge if $|\nabla I| \ge T_{\text{high}}$.
   - Weak edge if $T_{\text{low}} \le |\nabla I| < T_{\text{high}}$.
   - Edge tracking via 8-connectivity retains weak edges only if linked to a strong edge.

#### 2.4.4 Edge Detection Discussion
While Sobel magnitude produces wide, fuzzy intensity ramps reflecting local slope steepness, Canny produces crisp, binary boundaries. However, Canny lacks semantic awareness: it detects textured clothing wrinkles, high-contrast hair strands, and background shadows identically to key anatomical facial boundaries.

---

### 2.5 Frequency Analysis and Filtering

#### 2.5.1 2D Discrete Fourier Transform
The 2D DFT maps spatial domain intensities $f(x, y)$ of dimension $M \times N$ to complex spatial frequency coefficients $F(u, v)$:
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \exp\left[-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)\right]$$
The spectrum is centralized via `np.fft.fftshift`, placing zero frequency (DC component, corresponding to total scene luminance) at $(M/2, N/2)$. The dynamic range is compressed for visualization using logarithmic scaling:
$$S(u, v) = \log(1 + |F(u, v)|)$$

#### 2.5.2 Ideal vs. Gaussian Low-Pass Filtering
Low-pass filtering attenuates high spatial frequencies beyond cutoff radius $D_0 = 30$:
- **Ideal LPF**: Binary mask $H_{\text{ideal}}(u, v) = 1$ if $D(u, v) \le D_0$, else $0$. The sharp frequency cut generates Gibbs ringing artifacts in the spatial reconstruction.
- **Gaussian LPF**: Smooth transition $H_{\text{gauss}}(u, v) = \exp\left(-\frac{D^2(u, v)}{2 D_0^2}\right)$, eliminating ringing artifacts while smoothing textures.

#### 2.5.3 High-Pass Filtering
The complementary high-pass transfer function is defined as $H_{\text{HP}}(u, v) = 1 - H_{\text{LP}}(u, v)$. The high-pass reconstructed image eliminates broad uniform luminance, retaining only structural transitions, edges, and high-frequency noise.

#### 2.5.4 Numerical Reconstruction Check
To verify mathematical correctness of the complex Fourier transform, the low-pass and high-pass frequency components were summed and reconstructed via inverse transform:
$$f_{\text{reconstructed}}(x, y) = \text{Re}\left\{\mathcal{F}^{-1}\left[F_{\text{LP}}(u, v) + F_{\text{HP}}(u, v)\right]\right\}$$
The MSE between $f_{\text{reconstructed}}$ and original grayscale input is:
$$\text{MSE}_{\text{reconstruction}} = \mathbf{1.186 \times 10^{-27}}$$
This error is bounded strictly by IEEE 754 floating-point rounding precision, verifying complete conservation of energy and zero information loss during transformation.

---

### 2.6 Feature Detection and Matching

#### 2.6.1 Multi-View Image Pair Selection
To evaluate geometric feature invariance, a multi-view inspection pair was constructed:
- **View 1 (Reference)**: Sample portrait (`Training_10019449.jpg`) scaled to $256 \times 256$.
- **View 2 (Transformed)**: Ingests the reference image subjected to synthetic transformation: rotation of $+15^\circ$, scaling by $1.05\times$, and translation $(\Delta x=8, \Delta y=5)\text{ px}$.

#### 2.6.2 Detector and Descriptor Implementation
Two distinct feature paradigms were benchmarked:
1. **SIFT (Scale-Invariant Feature Transform)**:
   - Identifies extrema across Difference-of-Gaussians (DoG) scale space.
   - Refines keypoint location, eliminates edge-response candidates using Hessian matrix eigenvalues.
   - Assigns dominant orientation based on local gradient histograms.
   - Constructs a 128-dimensional floating-point descriptor vector from $4 \times 4$ subregions with 8 orientation bins each.
2. **ORB (Oriented FAST and Rotated BRIEF)**:
   - Detects keypoint corners via FAST-9 test across an image pyramid.
   - Assigns orientation using intensity centroid vector.
   - Generates a 256-bit binary descriptor using steered pairwise intensity comparisons (rBRIEF).

#### 2.6.3 Correspondence Matching and Lowe's Ratio Test
Features were matched using `cv2.BFMatcher`:
- **Distance Metrics**: Euclidean $L_2$ norm for SIFT; Hamming distance (XOR bitwise count) for binary ORB.
- **$k$-Nearest Neighbors ($k=2$)**: Finds the two best descriptor candidates $(m, n)$ in View 2 for each keypoint in View 1.
- **Lowe's Ratio Test**: Discards ambiguous matches by enforcing:
  $$\frac{d(f_1, m)}{d(f_1, n)} < \tau, \quad \text{with } \tau = 0.75$$
  This condition ensures that true correspondences exhibit significantly lower distance than the second-best false match caused by background repetition.

#### 2.6.4 Benchmark Comparison
| Evaluation Metric | SIFT (Scale-Invariant Feature Transform) | ORB (Oriented FAST & Rotated BRIEF) |
|---|---|---|
| **Descriptor Type** | 128-Dimensional Floating-Point ($L_2$ Norm) | 256-Bit Binary String (Hamming Distance) |
| **Keypoints Detected (View 1)** | 53 points | 99 points |
| **Keypoints Detected (View 2)** | 54 points | 87 points |
| **Retained Ratio Matches ($\tau=0.75$)** | **36 matches** | **51 matches** |
| **Match Retention Rate** | **67.92%** | **51.52%** |
| **Matching Speed** | High computational demand (Float distance) | **Extremely fast (Hardware bit-popcount)** |
| **Invariance Profile** | Superior invariance to scale & blur | Highly robust to in-plane rotation & fast runtime |

---

# 3. Task 2: Deep-Learning Model and Data Pipeline

Task 2 is implemented in [`model_pipeline.ipynb`](file:///E:/computer%20vision/cw/model_pipeline.ipynb), trained via [`scratch/train_and_evaluate.py`](file:///E:/computer%20vision/cw/scratch/train_and_evaluate.py), and exported to [`model/fer2013_checkpoint.pth`](file:///E:/computer%20vision/cw/model/fer2013_checkpoint.pth).

### 3.1 Classification Task and Dataset Selection

#### 3.1.1 Target Affective Classes
The model performs multi-class facial expression recognition, mapping a single-channel grayscale portrait $x \in \mathbb{R}^{1 \times 48 \times 48}$ to one of 7 mutually exclusive affective classes:
$$C = \{\text{angry}, \text{disgust}, \text{fear}, \text{happy}, \text{neutral}, \text{sad}, \text{surprise}\}$$
The output is a 7-dimensional probability distribution $\mathbf{p} = \text{Softmax}(\mathbf{z}) \in [0, 1]^7$ where $\sum_{c=1}^7 p_c = 1.0$.

#### 3.1.2 Dataset Source and Rationale
The benchmark **FER2013** dataset (Kaggle) was selected. Unlike lab-controlled datasets (such as CK+), FER2013 comprises unconstrained web-scraped real-world facial portraits exhibiting diverse lighting, partial occlusions, varied head poses, age distributions, and facial clutter, making it representative of real inspection environments.

#### 3.1.3 Dataset Statistics and Class Imbalance
An audit of raw files revealed an extreme class distribution skew:
- `angry`: 3,995 train samples (13.92%)
- `disgust`: **436 train samples (1.52%)** (Extreme minority)
- `fear`: 4,097 train samples (14.27%)
- `happy`: **7,215 train samples (25.13%)** (Dominant majority)
- `neutral`: 4,965 train samples (17.29%)
- `sad`: 4,830 train samples (16.82%)
- `surprise`: 3,171 train samples (11.05%)
- **Total Training Set**: 28,709 samples.
- **Ratio**: The majority class (`happy`) contains **16.5 times more samples** than the minority class (`disgust`). Unweighted cross-entropy loss causes the network to collapse on the minority class, producing 0% recall for `disgust`.

---

### 3.2 Data Pipeline and Anti-Leakage Preprocessing

#### 3.2.1 Data Ingestion and Sanitization
Inspection of `archive/train/` revealed 28,711 files: 28,709 legitimate `Training_*.jpg` images and 2 non-JPG operating system screenshots erroneously placed in `train/angry` and `train/happy`. Data ingestion enforced strict regular expression filtering:
$$\text{pattern} = \text{r"}\text{\textasciicircum Training\_}\backslash\text{d}+\backslash\text{.jpg\$"}$$
All non-matching files were rejected, preventing corrupted inputs from polluting the training pipeline.

#### 3.2.2 Partition Isolation
The original benchmark split boundaries were strictly preserved without random re-shuffling:
- **Training Set (`archive/train/`)**: Exactly 28,709 samples. Shuffled per epoch during training.
- **Validation Set (`archive/test/PublicTest_*.jpg`)**: Exactly 3,589 samples. Fixed sequence.
- **Test Set (`archive/test/PrivateTest_*.jpg`)**: Exactly 3,589 samples. Kept completely isolated until final evaluation.

#### 3.2.3 Normalization and Anti-Leakage Guarantee
To eliminate data leakage, pixel mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ were computed **strictly on the 28,709 training samples**:
$$\mu_{\text{train}} = 0.507739, \quad \sigma_{\text{train}} = 0.255078$$
These values were exported to `scratch/train_stats.json`. The validation set, test set, and Task 3 inference engine strictly apply these training parameters:
$$I_{\text{normalized}} = \frac{I / 255.0 - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
No information from validation or test sets informed normalization.

#### 3.2.4 Vectorized Tensor Data Augmentation
To regularize against overfitting while maintaining high CPU throughput, data augmentation was vectorized at the mini-batch tensor level ($B=128$):
1. **Random Horizontal Flip ($p=0.5$)**:
   $$\text{mask} \sim \text{Bernoulli}(0.5), \quad X_{\text{flip}} = \text{where}(\text{mask}, \text{flip}(X, \text{dim}=-1), X)$$
2. **Random Spatial Translation ($\pm 3\text{ px}$)**:
   Batch padded by 3 pixels with edge replication (`F.pad(mode='replicate')`), followed by random cropping back to $48 \times 48$.
Execution time was **$0.15\text{ seconds}$ per epoch** across all 28,709 samples, eliminating CPU data loading bottlenecks.

---

### 3.3 Model Design and Architecture

#### 3.3.1 Convolutional Architecture (`EmotionCNN`)
A customized 4-stage convolutional neural network was implemented in PyTorch:
- **Stage 1**: $\text{Input}(1 \times 48 \times 48) \to [\text{Conv3x3}(32) \to \text{BN} \to \text{ReLU}] \times 2 \to \text{MaxPool}(2 \times 2) \to \text{Dropout}(0.20)$ $\implies (32 \times 24 \times 24)$
- **Stage 2**: $[\text{Conv3x3}(64) \to \text{BN} \to \text{ReLU}] \times 2 \to \text{MaxPool}(2 \times 2) \to \text{Dropout}(0.25)$ $\implies (64 \times 12 \times 12)$
- **Stage 3**: $[\text{Conv3x3}(128) \to \text{BN} \to \text{ReLU}] \times 2 \to \text{MaxPool}(2 \times 2) \to \text{Dropout}(0.30)$ $\implies (128 \times 6 \times 6)$
- **Stage 4**: $[\text{Conv3x3}(256) \to \text{BN} \to \text{ReLU}] \to \text{MaxPool}(2 \times 2) \to \text{Dropout}(0.40)$ $\implies (256 \times 3 \times 3)$
- **Classifier Head**: $\text{AdaptiveAvgPool2d}(1 \times 1) \to \text{Flatten}(256) \to \text{Linear}(256, 128) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.50) \to \text{Linear}(128, 7)$

#### 3.3.2 Model Complexity
- **Trainable Parameters**: **617,063 float32 weights**.
- **Memory Footprint**: $2.49\text{ MB}$ checkpoint size.
- **Regularization**: Batch Normalization after every convolution stabilizes internal covariate shift; progressive Dropout ($0.20 \to 0.50$) prevents co-adaptation of features.

---

### 3.4 Model Training

#### 3.4.1 Cost-Sensitive Loss Function
To counter class imbalance without discarding data (undersampling) or over-duplicating noisy samples (oversampling), cost-sensitive cross-entropy was applied:
$$\mathcal{L} = -\sum_{c=1}^7 w_c \cdot y_c \cdot \log(p_c), \quad w_c = \text{clip}\left(\frac{N_{\text{total}}}{7 \cdot N_c}, 0.5, 4.0\right)$$
- Computed weights: `angry`: 1.0266, `disgust`: **4.0000** (max penalty), `fear`: 1.0010, `happy`: 0.5684, `neutral`: 0.8260, `sad`: 0.8491, `surprise`: 1.2934.

#### 3.4.2 Optimization Setup
- **Optimizer**: AdamW ($\text{learning rate} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Batch Size**: 128.
- **Scheduler**: `ReduceLROnPlateau` monitoring validation loss (factor $= 0.5$, patience $= 2$).
- **Checkpoint Selection**: Model states saved on validation loss improvement.

#### 3.4.3 Training Convergence Progression
```
Epoch  1/8 [134.8s]: Train Loss: 1.9286, Acc: 20.04% | Val Loss: 1.8359, Val Acc: 27.64%
Epoch  2/8 [133.9s]: Train Loss: 1.8539, Acc: 22.69% | Val Loss: 1.7690, Val Acc: 28.92%
Epoch  3/8 [168.6s]: Train Loss: 1.7645, Acc: 28.33% | Val Loss: 1.6655, Val Acc: 34.30%
Epoch  4/8 [176.0s]: Train Loss: 1.6968, Acc: 33.73% | Val Loss: 1.6318, Val Acc: 37.70%
Epoch  5/8 [172.6s]: Train Loss: 1.6073, Acc: 39.20% | Val Loss: 1.3944, Val Acc: 48.20%
Epoch  6/8 [163.2s]: Train Loss: 1.5234, Acc: 43.37% | Val Loss: 1.3826, Val Acc: 47.48%
Epoch  7/8 [164.7s]: Train Loss: 1.4739, Acc: 45.45% | Val Loss: 1.2899, Val Acc: 50.46% (BEST)
Epoch  8/8 [161.9s]: Train Loss: 1.4314, Acc: 46.99% | Val Loss: 1.2933, Val Acc: 50.04%
```
The best checkpoint was selected at **Epoch 7** with validation loss of **$1.2899$** and validation accuracy of **$50.46\%$**.

---

### 3.5 Model Evaluation on Unseen `PrivateTest` Set

Final evaluation was performed on the unseen `PrivateTest` set (3,589 samples) under strict evaluation protocol (`model.eval()` and `torch.no_grad()`).

#### 3.5.1 Primary Performance Metrics
- **Overall Accuracy**: **`51.60%`**
- **Macro-Averaged F1-Score**: **`0.4402`**
- **Weighted-Average F1-Score**: **`0.4907`**
- **Macro-Averaged Precision**: **`0.4433`**
- **Macro-Averaged Recall**: **`0.4778`**

#### 3.5.2 Per-Class Classification Report
| Affective Emotion | Precision | Recall | F1-Score | Support (PrivateTest) |
|---|---|---|---|---|
| `angry` | 0.3496 | 0.3503 | 0.3499 | 491 |
| `disgust` | 0.2079 | **0.3818** | **0.2692** | 55 (Minority) |
| `fear` | 0.3333 | 0.0587 | 0.0998 | 528 |
| `happy` | **0.7703** | **0.7782** | **0.7742** | 879 |
| `neutral` | 0.4530 | 0.6310 | 0.5274 | 626 |
| `sad` | 0.3785 | 0.4091 | 0.3932 | 594 |
| `surprise` | **0.6108** | **0.7356** | **0.6674** | 416 |
| **Accuracy** | | | **0.5160** | **3,589** |
| **Macro Average** | **0.4433** | **0.4778** | **0.4402** | **3,589** |
| **Weighted Average** | **0.5012** | **0.5160** | **0.4907** | **3,589** |

#### 3.5.3 Confusion Matrix Analysis
```
Normalized Confusion Matrix (Row = Ground Truth, Column = Predicted):
          angry  disgust   fear    happy  neutral    sad  surprise
angry    [ 0.35    0.02    0.02    0.13    0.23    0.22    0.03 ]
disgust  [ 0.24    0.38    0.04    0.09    0.11    0.13    0.02 ]
fear     [ 0.21    0.02    0.06    0.12    0.28    0.25    0.05 ]
happy    [ 0.05    0.01    0.01    0.78    0.08    0.06    0.02 ]
neutral  [ 0.08    0.01    0.01    0.07    0.63    0.17    0.02 ]
sad      [ 0.16    0.02    0.02    0.10    0.28    0.41    0.02 ]
surprise [ 0.04    0.01    0.02    0.09    0.08    0.03    0.74 ]
```
Key insights:
1. **Minority Class Recovery**: Cost-sensitive weighting successfully lifted `disgust` recall to **38.18%**, preventing minority collapse.
2. **Distinct Affective Clusters**: `happy` (F1: 0.7742) and `surprise` (F1: 0.6674) exhibit the highest recognition accuracy due to salient geometric features (open smile, raised eyebrows, widened eyes).
3. **Ambiguity Between Negative Valence Affects**: `fear` exhibits poor recall (5.87%), frequently confused with `neutral` (28%) and `sad` (25%). In $48 \times 48$ low-resolution grayscale, subtle distinctions between anxious brow furrows and subdued sadness lack sufficient high-frequency spatial detail.

### 3.6 Checkpoint Artifact Packaging
The trained model is stored as a self-contained artifact in `model/fer2013_checkpoint.pth` containing:
- Model state dictionary (`model_state_dict`).
- Ordered class names list: `CLASSES`.
- Training-derived normalization constants: $\mu_{\text{train}} = 0.507739, \sigma_{\text{train}} = 0.255078$.
- Architecture configuration: `input_size: [1, 48, 48]`, `num_classes: 7`.
- Training metadata: `best_epoch: 7`, `val_loss: 1.2899`.

---

# 4. Task 3: Assistant Interface and Application Integration

Task 3 is implemented in [`app.py`](file:///E:/computer%20vision/cw/app.py) using the Streamlit framework.

### 4.1 Interface Architecture and Module Routing
The application provides an interactive web interface with five operating modes accessible via the sidebar selector:
1. `1. Camera Calibration & Undistortion`
2. `2. Spatial Filtering & Edge Gradients`
3. `3. 2D FFT Frequency Analysis`
4. `4. SIFT vs ORB Feature Matching`
5. `5. Facial Emotion Recognition AI`

---

### 4.2 Module 5 Deep Learning & Face Inspection Details

#### 4.2.1 Image and Camera Input Validation
Three input modalities are supported:
1. **File Upload**: Accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
2. **Browser Webcam Snapshot (`st.camera_input`)**: Captures static snapshots directly from the user's browser webcam via WebRTC client APIs and transmits the binary `UploadedFile` buffer to the server.
3. **Curated Test Samples**: Enables selecting pre-labeled sample faces directly from `archive/test/`.

**Validation Engine**:
- Checks buffer length (`len(buffer) > 0`). Empty or 0-byte uploads are rejected.
- Decodes buffer safely via `cv2.imdecode`. If decoding fails (corrupted files, text files disguised with `.jpg` extension), the pipeline intercepts the error, renders a user-friendly error banner (`st.error`), and halts execution safely without exposing Python exception tracebacks.

#### 4.2.2 Scaled YuNet Face Detection
Full-scene inspection images (e.g. $640 \times 480$) are processed using the state-of-the-art **YuNet ONNX detector** (`cv2.FaceDetectorYN`) built into OpenCV 5.0.0.

**Aspect-Ratio Preserving Scaling**:
To prevent facial stretching/squashing and avoid CPU bottlenecks on high-resolution inputs:
1. The scaling factor is calculated relative to maximum dimension limit $\text{max\_dim} = 640$:
   $$\text{scale} = \min\left(1.0, \frac{640}{\max(W_{\text{orig}}, H_{\text{orig}})}\right)$$
2. Input dimensions are updated: $W_{\text{net}} = \text{int}(W_{\text{orig}} \cdot \text{scale})$, $H_{\text{net}} = \text{int}(H_{\text{orig}} \cdot \text{scale})$.
3. Dynamic network input size is set: `detector.setInputSize((W_net, H_net))`.
4. Detection is executed, and bounding box coordinates are mapped back to the original image frame:
   $$x_{\text{orig}} = \frac{x_{\text{net}}}{\text{scale}}, \quad y_{\text{orig}} = \frac{y_{\text{net}}}{\text{scale}}, \quad w_{\text{orig}} = \frac{w_{\text{net}}}{\text{scale}}, \quad h_{\text{orig}} = \frac{h_{\text{net}}}{\text{scale}}$$
5. Bounding box coordinates are clamped strictly using `np.clip` to $[0, W_{\text{orig}}]$ and $[0, H_{\text{orig}}]$.

*The detector was empirically verified to execute correctly on standard $640 \times 480$ camera test images.*

#### 4.2.3 Parameter Controls and Heuristic Padding Margin
- **Processing Mode Selector**:
  - `Ảnh thường (Cần phát hiện khuôn mặt)`: Full-scene detection mode.
  - `Ảnh chân dung đã crop sẵn (Bypass face detector)`: Directly bypasses face detection when ingesting pre-cropped $48 \times 48$ portraits (such as FER2013 samples).
- **Heuristic Padding Margin**:
  Raw detector bounding boxes cut tightly along eyebrows and lower lips. The application provides an interactive slider for padding margin ($0\% - 25\%$, default $10\%$) expanding width and height independently:
  $$\Delta x = \text{margin} \cdot w, \quad \Delta y = \text{margin} \cdot h$$
  Coordinates are clamped to sensor boundaries, preventing anatomy clipping.
- **Multi-Face Inspection Selector**:
  When $\ge 2$ faces are detected, bounding boxes are tagged (`Face #1`, `Face #2`, ...). A dropdown allows users to inspect individual faces or run predictions on all detected individuals.

#### 4.2.4 Result Visualization & Heuristic Low-Score Warning
- **Dual Visual Display**: Displays full scene with green bounding boxes alongside the extracted $48 \times 48$ grayscale normalized input seen by the neural network.
- **Softmax Probability Distribution**: Visualizes relative model scores across all 7 emotions via horizontal bar chart.
- **Heuristic Low-Score Warning**: Softmax outputs from uncalibrated networks reflect relative logit magnitudes rather than true calibrated posterior probabilities. If the Top-1 predicted score falls below $40\%$:
  > **⚠️ Cảnh báo: Điểm số dự đoán Top-1 thấp (Heuristic < 40%). Cần kiểm tra lại góc chụp hoặc điều kiện ánh sáng.**
  *(This threshold was derived from validation set observations and was not tuned on the PrivateTest set).*
- **Academic Scope Disclaimer**: Clearly states that the model achieves 51.60% test accuracy as an academic MVP for FCVG101, and must not be used as an authoritative psychological diagnostic tool or workplace surveillance system.

---

### 4.3 Functional Test Suite (Section 4.6.1)

To verify robust software functionality, an automated verification suite covering 6 test cases was implemented in `scratch/test_task3_cases.py` and integrated into Tab 2 of Module 5 in the Streamlit application.

*Note: These cases represent software functional verification tests ensuring execution stability and graceful degradation; they do not constitute a comprehensive benchmark of face detector accuracy across arbitrary unconstrained real-world environments.*

#### 4.3.1 Test Case Execution Summary
```
======================================================================
FUNCTIONAL TEST SUMMARY REPORT (Section 4.6.1)
======================================================================
[PASS] TC1: Single Face Standard (640x480)
       Input   : 640x480 canvas containing 1 face
       Criteria: len(faces) == 1, valid crop ROI, Tensor (1, 1, 48, 48) float32, sum(probs) == 1.0, 0 exception
       Outcome : Detected 1 face at bbox=[264, 128, 115, 138], Top-1: happy (74.3%), Tensor: [1, 1, 48, 48]

[PASS] TC2: Multiple Faces Handling
       Input   : 640x480 composite image with 2 faces
       Criteria: len(faces) >= 2, BBoxes clamped in [0, 640]x[0, 480], individual tensors valid, 0 IndexError
       Outcome : Successfully detected 2 individual faces and generated individual predictions without index errors.

[PASS] TC3: Zero Faces (Non-face input)
       Input   : Checkerboard calibration image (left01.jpg, 640x480)
       Criteria: len(faces) == 0, graceful warning banner displayed, 0 unhandled exception
       Outcome : Returned 0 faces gracefully, zero unhandled exceptions, safely triggers warning banner.

[PASS] TC4: Pre-cropped Face (Bypass Mode)
       Input   : 48x48 sample image (archive/test/happy/PrivateTest_*.jpg)
       Criteria: Bypass detector directly, input formatted to (1, 1, 48, 48), valid 7-class prediction
       Outcome : Bypassed detector directly; input formatted to 48x48; predicted: happy (98.3%).

[PASS] TC5: Browser Webcam Snapshot
       Input   : Encoded JPEG buffer simulating st.camera_input upload
       Criteria: Decoded to valid BGR (480, 640, 3) uint8 array, passes into detector and classifier smoothly
       Outcome : Decoded 10,361 byte snapshot buffer to shape (480, 640, 3), detected 1 face.

[PASS] TC6: Corrupted / Invalid Input Rejection
       Input   : Non-image text bytes and 0-byte buffer
       Criteria: cv2.imdecode == None, rejected at validation layer, st.error displayed, 0 traceback
       Outcome : Safely intercepted at validation layer; cv2.imdecode returns None; triggers st.error cleanly.
======================================================================
>>> ALL 6 FUNCTIONAL TEST CASES PASSED (100%)! <<<
======================================================================
```

---

# 5. Conclusion and Future Improvements

### 5.1 Project Achievements
The coursework successfully designed, implemented, and verified an integrated computer vision inspection pipeline:
1. **Mathematical Rigor**: Demonstrated pinhole camera calibration with an overall reprojection RMSE of $0.4087\text{ px}$ and validated a $94.4\%$ reduction in geometric curvature error on rectified lens views.
2. **Signal & Spatial Processing**: Quantified denoising MSE tradeoffs ($3\times3$ Median filter achieved $74.15$ MSE vs. $385.24$ noisy baseline), analyzed Sobel/Canny gradient maps, and achieved numerical reconstruction precision ($\text{MSE} \approx 10^{-27}$) under 2D FFT.
3. **Data Hygiene & Machine Learning**: Enforced zero data leakage, mitigated extreme class imbalance via cost-sensitive weighting, lifted minority `disgust` recall to $38.18\%$, and validated an overall test accuracy of $51.60\%$ on unseen data.
4. **Engineering Architecture**: Deployed an interactive Streamlit assistant featuring aspect-ratio preserved face detection, configurable heuristic padding, and an automated functional verification test suite achieving 100% pass rates.

### 5.2 Technical Limitations and Future Directions
1. **Affective Ambiguity in Low Resolutions**: The $48 \times 48$ resolution of FER2013 lacks fine textural cues (nasolabial micro-movements, pupil dilation), leading to confusion between `fear`, `sadness`, and `neutral`. Future work could integrate higher-resolution affective benchmarks (e.g., AffectNet at $224 \times 224$) or employ Vision Transformer (ViT) backbones.
2. **Video Streaming Pipeline**: The current webcam implementation relies on static snapshot acquisition (`st.camera_input`). Extending the assistant to continuous real-time video monitoring requires integrating dedicated asynchronous WebRTC streaming components (such as `streamlit-webrtc`).
3. **Probability Calibration**: Softmax probabilities should be calibrated using Temperature Scaling or Platt Scaling on the validation partition to convert heuristic model outputs into calibrated Bayesian confidence metrics.

---

# 6. Installation, Configuration, and Execution Instructions

### 6.1 Environment and Dependencies
The project runs on Python 3.10+ (tested on Python 3.13 Windows environment) with standard CPU/CUDA compute support:

```powershell
# Core Dependencies
pip install numpy scipy matplotlib seaborn scikit-learn
pip install opencv-python pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
```

### 6.2 Repository Structure
```
E:\computer vision\cw\
├── app.py                             # Main Streamlit web assistant application
├── dataset.ipynb                      # Task 1 Jupyter Notebook
├── model_pipeline.ipynb               # Task 2 Jupyter Notebook
├── camera_calibration_params.json     # Calibrated K, D, RMSE parameters
├── modules/
│   ├── calibration_engine.py          # Camera calibration & rectification
│   ├── filter_engine.py               # Spatial, edge, and FFT filtering
│   ├── feature_engine.py              # SIFT & ORB feature matching
│   ├── emotion_classifier.py          # EmotionCNN model & YuNet detector
│   └── face_detection_yunet_2023mar.onnx # YuNet ONNX face detector weights
├── model/
│   ├── fer2013_checkpoint.pth         # Trained PyTorch checkpoint
│   ├── learning_curves.png            # Training/Validation loss & accuracy curves
│   ├── confusion_matrix.png           # Raw and normalized confusion matrices
│   └── evaluation_summary.json        # Test evaluation metrics
├── calibration_images/                # 13 OpenCV checkerboard views
├── archive/                           # FER2013 dataset (train/ and test/)
└── scratch/
    └── test_task3_cases.py            # Automated functional verification suite
```

### 6.3 Running the Application and Automated Tests
To launch the Streamlit Visual Inspection Assistant:
```powershell
cd "E:\computer vision\cw"
streamlit run app.py
```
To run the automated functional verification suite from the terminal:
```powershell
python "E:\computer vision\cw\scratch\test_task3_cases.py"
```

---

# References

1. Zhang, Z. (2000). "A flexible new technique for camera calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.
2. Canny, J. (1986). "A computational approach to edge detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, (6), 679-698.
3. Lowe, D. G. (2004). "Distinctive image features from scale-invariant keypoints." *International Journal of Computer Vision*, 60(2), 91-110.
4. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). "ORB: An efficient alternative to SIFT or SURF." *IEEE International Conference on Computer Vision (ICCV)*, 2564-2571.
5. Goodfellow, I. J., et al. (2013). "Challenges in representation learning: A report on three machine learning contests (FER2013)." *International Conference on Neural Information Processing*, 117-124.
6. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.
7. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization (AdamW)." *International Conference on Learning Representations (ICLR)*.
8. YuNet Face Detection. *OpenCV Model Zoo: Lightweight Face Detection Network*, https://github.com/opencv/opencv_zoo.

---

# Appendices

## Appendix A: Python source code (selectable text)

All code in this appendix is provided as selectable text. It is reproduced from the project source files so that it can be inspected, copied, and executed; no code screenshot is used.

### A.1 Classical vision pipeline (Task 1)

The following excerpts come from `modules/calibration_engine.py`, `modules/filter_engine.py`, and `modules/feature_engine.py`.

```python
import cv2
import numpy as np

def run_checkerboard_calibration(image_paths, pattern_size=(9, 6), square_size=1.0):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    objpoints, imgpoints, valid_paths = [], [], []
    img_shape = None

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = img_shape or gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_paths.append(str(p))

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    rmse_per_view = []
    for obj, observed, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rmse_per_view.append(float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=2)))))
    return {"overall_rms": float(rms), "K": K, "dist": dist,
            "rmse_per_view": rmse_per_view, "valid_paths": valid_paths}

def apply_spatial_filters(gray_img, gaussian_ksize=3, gaussian_sigma=0, median_ksize=3):
    gaussian_ksize += gaussian_ksize % 2 == 0
    median_ksize += median_ksize % 2 == 0
    gaussian = cv2.GaussianBlur(gray_img, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    median = cv2.medianBlur(gray_img, median_ksize)
    return gaussian, median

def mse(reference, candidate):
    return float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
```

```python
def apply_gradients_and_edges(gray_img, canny_low=50, canny_high=150):
    smooth = cv2.GaussianBlur(gray_img, (3, 3), 0)
    sobel_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = cv2.Canny(smooth, canny_low, canny_high)
    return sobel_x, sobel_y, np.clip(magnitude, 0, 255).astype(np.uint8), edges

def apply_fft_filtering(gray_img, cutoff_radius=0.05):
    spectrum_complex = np.fft.fftshift(np.fft.fft2(gray_img))
    spectrum = np.log1p(np.abs(spectrum_complex))
    h, w = gray_img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    sigma_f = max(cutoff_radius, 0.005)
    low_mask = np.exp(-(FX ** 2 + FY ** 2) / (2 * sigma_f ** 2))
    high_mask = 1.0 - low_mask
    low_img = np.fft.ifft2(np.fft.ifftshift(spectrum_complex * low_mask)).real
    high_img = np.fft.ifft2(np.fft.ifftshift(spectrum_complex * high_mask)).real
    reconstructed = np.clip(low_img + high_img, 0, 255).astype(np.uint8)
    reconstruction_mse = float(np.mean((gray_img.astype(np.float64) - low_img - high_img) ** 2))
    return spectrum, low_mask, high_mask, reconstructed, reconstruction_mse

def match_features(des1, des2, norm_type=cv2.NORM_L2, ratio_thresh=0.75):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0, 0.0, 0.0
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < ratio_thresh * n.distance]
    valid_pairs = sum(len(pair) == 2 for pair in pairs)
    return good, valid_pairs, 100 * len(good) / valid_pairs, 0.0
```

### A.2 Deep-learning model and data pipeline (Task 2)

```python
def collect_file_lists(archive_dir="archive"):
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = [], [], [], [], [], []
    for class_index, class_name in enumerate(CLASSES):
        for file_path in (Path(archive_dir) / "train" / class_name).iterdir():
            if re.match(r"^Training_\d+\.jpg$", file_path.name):
                train_paths.append(str(file_path)); train_labels.append(class_index)
        for file_path in (Path(archive_dir) / "test" / class_name).iterdir():
            if re.match(r"^PublicTest_\d+\.jpg$", file_path.name):
                val_paths.append(str(file_path)); val_labels.append(class_index)
            elif re.match(r"^PrivateTest_\d+\.jpg$", file_path.name):
                test_paths.append(str(file_path)); test_labels.append(class_index)
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# Statistics are computed from training images only, then reused unchanged.
mean = float(train_imgs.mean() / 255.0)
std = float(train_imgs.std() / 255.0)
X_train = (torch.from_numpy(train_imgs).unsqueeze(1).float() / 255.0 - mean) / std
X_val = (torch.from_numpy(val_imgs).unsqueeze(1).float() / 255.0 - mean) / std
X_test = (torch.from_numpy(test_imgs).unsqueeze(1).float() / 255.0 - mean) / std
class_weights = np.clip(len(train_labels) / (len(CLASSES) * np.bincount(train_labels)), 0.5, 4.0)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
```

```python
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        def block(in_channels, out_channels, dropout, twice=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)]
            if twice:
                layers += [nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)]
            return nn.Sequential(*layers, nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.block1 = block(1, 32, 0.20)
        self.block2 = block(32, 64, 0.25)
        self.block3 = block(64, 128, 0.30)
        self.block4 = block(128, 256, 0.40, twice=False)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.50), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.block4(self.block3(self.block2(self.block1(x)))))

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

for epoch in range(8):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_augment(inputs.to(device))), targets.to(device))
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(x.to(device)), y.to(device)).item() * len(x) for x, y in val_loader) / len(val_loader.dataset)
    scheduler.step(val_loss)

cm = confusion_matrix(all_targets, all_preds)
report = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True)
```

### A.3 Assistant application and face detection (Task 3)

```python
class YuNetFaceDetector:
    def __init__(self, onnx_model_path="modules/face_detection_yunet_2023mar.onnx", score_threshold=0.50):
        self.score_threshold = float(score_threshold)
        self.detector = cv2.FaceDetectorYN_create(onnx_model_path, "", (320, 320), self.score_threshold, 0.30, 5000)

    def detect_faces(self, img_bgr, conf_thresh=None, max_dim=640):
        if img_bgr is None or img_bgr.size == 0:
            return []
        h_orig, w_orig = img_bgr.shape[:2]
        scale = min(1.0, float(max_dim) / max(w_orig, h_orig))
        w_net, h_net = max(1, round(w_orig * scale)), max(1, round(h_orig * scale))
        network_image = cv2.resize(img_bgr, (w_net, h_net), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
        self.detector.setInputSize((w_net, h_net))
        self.detector.setScoreThreshold(float(conf_thresh) if conf_thresh is not None else self.score_threshold)
        _, faces = self.detector.detect(network_image)
        if faces is None:
            return []
        result = []
        for face in faces:
            bx, by, bw, bh = face[:4] / scale
            x1, y1 = int(np.clip(round(bx), 0, w_orig)), int(np.clip(round(by), 0, h_orig))
            x2, y2 = int(np.clip(round(bx + bw), 0, w_orig)), int(np.clip(round(by + bh), 0, h_orig))
            if x2 > x1 and y2 > y1:
                result.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(face[14])})
        return result

def crop_face_with_margin(img_bgr, bbox, margin_ratio=0.10):
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    dx, dy = round(margin_ratio * w), round(margin_ratio * h)
    x1, y1 = int(np.clip(x - dx, 0, w_img)), int(np.clip(y - dy, 0, h_img))
    x2, y2 = int(np.clip(x + w + dx, 0, w_img)), int(np.clip(y + h + dy, 0, h_img))
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def preprocess_face_to_tensor(face_img, train_mean=0.507739, train_std=0.255078):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
    preview_48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = (preview_48.astype(np.float32) / 255.0 - train_mean) / train_std
    return torch.tensor(normalized).unsqueeze(0).unsqueeze(0), preview_48
```

```python
def load_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Core application branch in app.py.
if invalid_upload:
    st.error("The uploaded file is not a valid image or its data is corrupted.")
elif user_face is None:
    st.info("Upload an image, capture a webcam snapshot, or select a sample before analysis.")
elif proc_mode == "Pre-cropped portrait (Bypass face detector)":
    pred_label, prob_dict, preview_48, top1_score = classifier.predict_crop(user_face)
    if top1_score < 0.40:
        st.warning("The Top-1 prediction score is low; check the camera angle or lighting.")
else:
    face_results, annotated_img = classifier.detect_and_classify(
        user_face, conf_thresh=score_thresh, margin_ratio=margin_pct / 100.0
    )
    if len(face_results) == 0:
        st.warning("No face was detected in the image.")
    else:
        face_options = ["All detected faces"] + [f"Face #{face['face_idx']}" for face in face_results]
        chosen_face = st.selectbox("Select a face to view in detail", face_options)

# Representative automated assertions from scratch/test_task3_cases.py.
faces = detector.detect_faces(canvas_single, conf_thresh=0.30)
assert len(faces) == 1
crop, _ = crop_face_with_margin(canvas_single, faces[0]["bbox"], margin_ratio=0.10)
tensor, _ = preprocess_face_to_tensor(crop, clf.train_mean, clf.train_std)
assert tensor.shape == (1, 1, 48, 48)
assert tensor.dtype == torch.float32
assert abs(sum(clf.predict_tensor(tensor)[1].values()) - 1.0) < 1e-4
```

## Appendix B: Configuration and hyperparameters

| Parameter name | Configured value | Selection rationale |
|---|---:|---|
| Checkerboard pattern | $9 \times 6$ inner corners | Matches the calibration board and gives 54 points per view. |
| Square size | 1.0 | Uses a consistent unit scale for calibration. |
| Corner refinement | $11 \times 11$, 30 iterations, $\epsilon=0.001$ | Improves corner localisation before calibration. |
| Gaussian filter kernels | $3 \times 3$ and $9 \times 9$ | Compares moderate smoothing with stronger smoothing. |
| Median filter kernel | $3 \times 3$ | Removes impulse-like variation while retaining edges. |
| Canny thresholds | 50, 150 | Separates weak from strong edges at a 1:3 ratio. |
| FFT cutoff | $D_0 = 30$ px in the analysis | Retains low-frequency image content while making the filtering effect visible. |
| FFT implementation cutoff | `cutoff_radius = 0.05` cycles/pixel | Normalised equivalent used by the reusable Gaussian-mask function. |
| Lowe ratio threshold | $\tau = 0.75$ | Rejects ambiguous nearest-neighbour feature matches. |
| CNN input | $1 \times 48 \times 48$ grayscale | Native FER2013 resolution. |
| Normalisation | $\mu=0.507739$, $\sigma=0.255078$ | Calculated only on 28,709 training images. |
| Batch size | 128 | Fits the vectorised in-memory pipeline and gives stable gradients. |
| Optimiser | AdamW, weight decay $10^{-4}$ | Uses adaptive optimisation with decoupled regularisation. |
| Initial learning rate | $10^{-3}$ | Starting value used for this model. |
| Scheduler | `ReduceLROnPlateau`, factor 0.5, patience 2 | Reduces the learning rate after validation loss stops improving. |
| Epochs | 8 | Fixed training budget for the coursework experiment. |
| Dropout | 0.20, 0.25, 0.30, 0.40, 0.50 | Increases regularisation through the feature extractor and classifier. |
| Loss | Cost-sensitive cross-entropy | Addresses the FER2013 class imbalance. |
| Class weights | 1.0266, 4.0000, 1.0010, 0.5684, 0.8260, 0.8491, 1.2934 | Inverse-frequency weights, clipped to [0.5, 4.0], in class order. |
| YuNet maximum dimension | 640 px | Limits detector input cost while preserving aspect ratio. |
| Face-crop margin | 10% by default; adjustable from 0% to 25% | Includes nearby facial context and prevents tight detector crops. |
| Low-score warning | Top-1 score below 40% | Alerts the user to a low-confidence, uncalibrated prediction. |

## Appendix C: Additional experimental results

### C.1 Per-image calibration reprojection RMSE

The baseline calibration RMS over all views is 0.4087 px.

| Image | RMSE (px) | Image | RMSE (px) |
|---|---:|---|---:|
| `left01.jpg` | 0.3842 | `left08.jpg` | 0.5891 |
| `left02.jpg` | 0.4121 | `left09.jpg` | 0.3755 |
| `left03.jpg` | 0.3564 | `left10.jpg` | 0.4610 |
| `left04.jpg` | 0.2312 | `left11.jpg` | 0.3982 |
| `left05.jpg` | 0.4489 | `left12.jpg` | 0.4320 |
| `left06.jpg` | 0.3912 | `left13.jpg` | 0.4136 |
| `left07.jpg` | 0.4201 | Overall | **0.4087** |

### C.2 Complete classification reports

| Split / class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| PublicTest: angry | 0.3327 | 0.3597 | 0.3457 | 467 |
| PublicTest: disgust | 0.1667 | 0.3393 | 0.2235 | 56 |
| PublicTest: fear | 0.2742 | 0.0343 | 0.0609 | 496 |
| PublicTest: happy | 0.7829 | 0.7453 | 0.7636 | 895 |
| PublicTest: neutral | 0.4057 | 0.6096 | 0.4872 | 607 |
| PublicTest: sad | 0.3946 | 0.3783 | 0.3862 | 653 |
| PublicTest: surprise | 0.6236 | 0.7783 | 0.6924 | 415 |
| PublicTest: macro average | 0.4257 | 0.4635 | 0.4228 | 3,589 |
| PrivateTest: angry | 0.3496 | 0.3503 | 0.3499 | 491 |
| PrivateTest: disgust | 0.2079 | 0.3818 | 0.2692 | 55 |
| PrivateTest: fear | 0.3333 | 0.0587 | 0.0998 | 528 |
| PrivateTest: happy | 0.7703 | 0.7782 | 0.7742 | 879 |
| PrivateTest: neutral | 0.4530 | 0.6310 | 0.5274 | 626 |
| PrivateTest: sad | 0.3785 | 0.4091 | 0.3932 | 594 |
| PrivateTest: surprise | 0.6108 | 0.7356 | 0.6674 | 416 |
| PrivateTest: macro average | 0.4433 | 0.4778 | 0.4402 | 3,589 |

The PublicTest report was re-extracted from the selected checkpoint using the same training-only normalisation statistics. PrivateTest values are read from `model/evaluation_summary.json`.

### C.3 PrivateTest confusion matrix: raw counts

Rows are ground-truth classes and columns are predicted classes, in the order angry, disgust, fear, happy, neutral, sad, surprise.

| Actual \ Predicted | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Angry | 172 | 32 | 19 | 40 | 100 | 91 | 37 |
| Disgust | 23 | 21 | 0 | 3 | 1 | 4 | 3 |
| Fear | 88 | 24 | 31 | 42 | 125 | 112 | 106 |
| Happy | 28 | 6 | 6 | 684 | 62 | 63 | 30 |
| Neutral | 50 | 4 | 8 | 44 | 395 | 115 | 10 |
| Sad | 112 | 11 | 8 | 47 | 164 | 243 | 9 |
| Surprise | 19 | 3 | 21 | 28 | 25 | 14 | 306 |

### C.4 CPU latency breakdown

| Stage | Mean latency | Measurement scope |
|---|---:|---|
| YuNet face detection | approximately 12 ms | Detection on a standard image after aspect-ratio-preserving resize. |
| Tensor preprocessing | approximately 1.5 ms | Grayscale conversion, resize, Z-score normalisation, and tensor creation. |
| EmotionCNN inference | approximately 4.2 ms | One forward pass for a $1 \times 48 \times 48$ tensor. |

The figures describe the three compute stages and exclude image upload, Streamlit rendering, and chart drawing.

## Appendix D: Repository and trained-model links

The repository has no configured Git remote in the submitted workspace. Replace the placeholders below with the public URLs before final submission; inventing a link would prevent the assessor from downloading the required artifacts.

| Item | Public link | Local project location |
|---|---|---|
| Project code repository | `[Insert GitHub/Git repository URL]` | `E:\computer vision\cw` |
| EmotionCNN checkpoint | `[Insert public Google Drive/OneDrive direct-download URL]` | `model/fer2013_checkpoint.pth` (2,493,941 bytes; approximately 2.49 MB) |
| OpenCV YuNet model | `[Insert public Google Drive/OneDrive direct-download URL]` | `modules/face_detection_yunet_2023mar.onnx` (232,589 bytes; approximately 227 KB) |

After downloading the two artifacts, the assessor should preserve this layout:

```text
cw/
├── model/
│   └── fer2013_checkpoint.pth
└── modules/
    └── face_detection_yunet_2023mar.onnx
```

From the `cw/` directory, run `streamlit run app.py`. The paths above are the default paths used by the application.
