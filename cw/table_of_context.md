Table of Contents

1. Introduction
- 1.1 Project Background and Problem Statement
- 1.2 Project Objectives and Scope
- 1.3 System Overview
2. Task 1: Core Image Analysis Engine
- 2.1 Image Acquisition and Preparation
  - 2.1.1 Image Sources and Input Properties
  - 2.1.2 Grayscale Conversion and Data Types
- 2.2 Image Formation and Camera Calibration
  - 2.2.1 Pinhole Camera Model
  - 2.2.2 Checkerboard Image Collection and Corner Detection
  - 2.2.3 Camera Intrinsic Matrix and Distortion Coefficients
  - 2.2.4 Image Undistortion
  - 2.2.5 Reprojection Error and Results
- 2.3 Image Filtering and Noise Reduction
  - 2.3.1 Gaussian Noise Generation
  - 2.3.2 Gaussian Filtering: 3 × 3 and 9 × 9 Kernels
  - 2.3.3 Median Filtering
  - 2.3.4 Visual Comparison and MSE Evaluation
- 2.4 Image Gradients and Edge Detection
  - 2.4.1 Sobel X and Sobel Y
  - 2.4.2 Gradient Magnitude
  - 2.4.3 Canny Edge Detection
  - 2.4.4 Results and Discussion
- 2.5 Frequency Analysis and Filtering
  - 2.5.1 Fourier Transform and Spectrum Visualization
  - 2.5.2 Low-Pass Filtering
  - 2.5.3 High-Pass Filtering
  - 2.5.4 Inverse Transform and Reconstruction Check
  - 2.5.5 Results and Discussion
- 2.6 Feature Detection and Matching
  - 2.6.1 Image Pair Selection
  - 2.6.2 Selected Detector and Descriptor
  - 2.6.3 Feature Matching and Ratio Test
  - 2.6.4 Matching Results and Quantitative Analysis
- 2.7 Core Pipeline Structure and Limitations
3. Task 2: Deep-Learning Model and Data Pipeline
- 3.1 Classification Task and Dataset Selection
  - 3.1.1 Target Classes and Expected Outputs
  - 3.1.2 Dataset Source and Selection Rationale
  - 3.1.3 Dataset Statistics and Class Distribution
- 3.2 Data Pipeline
  - 3.2.1 Data Ingestion and Cleaning
  - 3.2.2 Training, Validation and Test Splits
  - 3.2.3 Resizing and Normalization
  - 3.2.4 Training Data Augmentation
- 3.3 Model Design
  - 3.3.1 Architecture and Selection Rationale
  - 3.3.2 Input and Output Specifications
  - 3.3.3 Training Strategy
- 3.4 Model Training
  - 3.4.1 Loss Function and Optimizer
  - 3.4.2 Hyperparameters and Experimental Setup
  - 3.4.3 Checkpoint Selection and Model Saving
- 3.5 Model Evaluation
  - 3.5.1 Training and Validation Curves
  - 3.5.2 Test Accuracy, Precision, Recall and F1-Score
  - 3.5.3 Confusion Matrix
  - 3.5.4 Prediction Examples and Error Analysis
- 3.6 Saved Model and Inference Pipeline
- 3.7 Model Limitations
4. Task 3: Assistant Interface and Application Integration
- 4.1 User Requirements and Application Workflow
- 4.2 Interface Design
  - 4.2.1 Image Input and Validation
  - 4.2.2 Task Selection and Parameter Controls
  - 4.2.3 Result Visualization
- 4.3 Integration of the Task 1 Pipeline
  - 4.3.1 Camera Calibration and Undistortion
  - 4.3.2 Filtering, Gradients and Edge Detection
  - 4.3.3 Frequency Analysis
  - 4.3.4 Feature Detection and Matching
- 4.4 Integration of the Task 2 Model
  - 4.4.1 Model Loading
  - 4.4.2 Input Preprocessing
  - 4.4.3 Predicted Class and Confidence Display
- 4.5 Error Handling and Input Validation
- 4.6 Application Testing and Demonstration
  - 4.6.1 Functional Test Cases and Results
  - 4.6.2 Interface Screenshots
  - 4.6.3 Integration Limitations
5. Conclusion and Future Improvements

6. Installation and Execution Instructions
- 6.1 Environment and Dependencies
- 6.2 Dataset and Model Access
- 6.3 Running the Application
References

Appendices
- Appendix A: Python Source Code — Selectable Text
- Appendix B: Configuration and Hyperparameters
- Appendix C: Additional Results
- Appendix D: Repository and Trained Model Links