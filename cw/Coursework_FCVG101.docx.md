

| Module: FCVG101 Introduction to Computer Vision | Coursework |
| :---- | :---- |
| **Contribution: 60% of the grade** | Coursework submission: A single PDF report submitted on CMS. |
| Module leader: | Due date: |
| **Approximate time to complete the coursework: 60 hours** |  |
| Learning outcomes: | 1\. Understand and apply image formation, camera geometry and camera calibration (LO1). 2\. Implement and evaluate linear filtering, frequency analysis, image gradients and interest-point detection (LO2). 3\. Build and evaluate convolutional neural networks and representation learning for vision (LO3). 4\. Apply multi-view geometry, stereo vision, structure-from-motion and optical flow to recover depth and motion (LO4). |

**Plagiarism**

Plagiarism is presenting somebody else's work as one's own. It includes copying information directly from online resources or books without referencing the material, submitting joint coursework as an individual effort; copying another student's coursework, purchasing coursework from someone else and submitting it as own work. Suspected plagiarism will be investigated and if found to have occurred will be dealt with according to the procedures set by the University. All material copied or amended from any source must be referenced correctly according to specific reference style.

**Coursework will be submitted for electronic plagiarism checks.**

**Coursework Submission Requirements**

* An electronic copy of the coursework should be uploaded on CMS before the deadline  
* The last uploaded version will be the one that is marked  
* The format of the coursework is a single PDF document containing all answers including: Python source code; Generated images; Obtained numerical results  
* The limit of the file size is 5 MB  
* The PDF document should be virus-free, not protected by a password or corrupted, otherwise it will be treated as not submitted  
* Grade and feedback will be available on CMS

In order to be marked, the source code needs to meet the following requirements:

1. To be selectable as text  
2. To be error free (except errors due to indentation)  
3. To generate the required visualization and results

**Coursework Specification:**

**Scenario:**

You will take the role as a Computer Vision engineer working on an outsource project. The project is an MVP and about developing a computer vision system working as a visual inspection assistant such as a warehouse manager assistant, an organic farm assistant, ...

All minimum requirements for the computer vision system are as below:

* (a) The computer vision system can acquire images from a camera (or a provided dataset) and understand them through the image-formation and camera-geometry pipeline (pinhole camera model, camera calibration).  
* (b) The computer vision system can process images with classical computer-vision techniques (linear filtering, image gradients, frequency analysis, feature detection and matching) and bring out quantitative results.  
* (c) The computer vision system can also do some tasks such as classification, prediction, depth/motion measurement with a trained deep-learning model.

In this MVP, you can choose any kind of dataset for finishing requirements b and c.

**Task 1**

Develop the core image analysis engine of the system: the imaging and classical computer-vision pipeline. You can use OpenCV, NumPy, SciPy, scikit-image, ... or implement the techniques from scratch.

**Task 2**

Develop at least 1 deep-learning model and 1 data pipeline for requirements b and c.

**Task 3**

Add an assistant feature (application integration / tool use) to your computer vision system: a user interface that runs the Task 1 pipeline and the Task 2 model and returns visualized results.

**Guidance**

**Task 1: Core Image Analysis Engine**

* Toolkit choice: You have the freedom to choose. OpenCV \+ NumPy is the standard. Implementing key techniques from scratch (e.g., Harris corner detector, SIFT-style matching, camera calibration by DLT) is more challenging but demonstrates deep technical skill (recommended for higher score).  
* Implementation: The core deliverable is a working image analysis pipeline (e.g., a Python script or notebook) that ingests an image or camera feed and produces calibrated, filtered, and feature-based outputs.  
* Camera calibration (Requirement a): calibrate a camera with a checkerboard (cv2.calibrateCamera), report the intrinsic matrix K and distortion coefficients, and undistort images.  
* Filtering & gradients: apply correlation/convolution filters (box, Gaussian, median), denoise a noisy image, compute gradients (Sobel, Canny) and frequency-domain filtering.  
* Features: detect interest points (Harris, SIFT, ORB) and match them across two views with the ratio test.

Minimum requirement (a): the pipeline must ingest images, report camera intrinsics and produce correct filtered / feature visualizations.

**Task 2: Deep-Learning Model and Data Pipeline**

This is the core learning component that validates technical CV skills.

* Dataset selection: Choose a dataset appropriate for both data analysis/results and a specific task (classification or prediction). Example: a public dataset of objects on Kaggle.  
* Data pipeline: A clear, documented sequence of steps to take raw data and prepare it for the model. This typically includes: Data ingestion (loading the data); Data cleaning (handling missing values, outliers, bad images); Feature engineering / augmentation (rotation, scaling, noise injection); Data transformation/scaling (resizing, normalization, train/validation split).  
* DL model: Implement at least one model (PyTorch or TensorFlow). For classification: a CNN (e.g., a simple Conv→ReLU→Pool stack, VGG-style, or a pretrained backbone with transfer learning). For prediction/regression: e.g., depth from stereo disparity, optical-flow magnitude, or keypoint-based pose estimation.  
* Deliverable: You must provide the Python code for the data pipeline and model training/evaluation, and crucially, make the trained model accessible to Task 3 (saved model artifact).

**Task 3: Assistant Feature (Application Integration / Tool Use)**

This task links your image analysis engine (Task 1\) to your deep-learning model (Task 2).

* Concept: Application integration (or tool use) allows the system to recognize the user's intent that requires a specific module — running the filtering/feature pipeline or calling the trained model — and present the result back to the user.  
* Recommended implementation: A small GUI web app using Streamlit or Gradio where the manager can: upload an image; choose a mode (e.g., calibrate / filter / detect features / classify / measure); see the pipeline visualization (undistorted image, edges, keypoints, matches, disparity map, Grad-CAM heatmap) and the model's output (predicted class with confidence, measured depth, etc.).

**Submission Requirements**

Your submission is PDF format file, should include the following:

* Project Code Repository: All source code (Python, configuration files, etc.).  
* Trained Model Artifact: The saved, trained model file (shared link).  
* Project Report/ReadMe File: A document detailing: the chosen dataset and justification; the complete data pipeline steps; the implemented model, hyperparameters, and evaluation results; how the application integration / tool use feature is implemented; instructions for running the entire system.

**Scoring Schema**

| Criteria | Max Points | Description and Focus Areas |
| :---- | :---- | :---- |
| **I. Task 1: Core Image Analysis Engine** | **20** | Functional and Image-Processing Depth |
| **1.1 Image formation & camera calibration** | **5** | Correct use of the pinhole camera model; checkerboard calibration; reported intrinsic matrix and undistortion (Requirement a). |
| **1.2 Filtering & edge processing** | **5** | Appropriate filters (box/Gaussian/median, gradients, frequency domain) and correct use of OpenCV functions. |
| **1.3 Feature detection & matching** | **5** | Harris/SIFT/ORB keypoints detected and matched across views with the ratio test; quality of matching results. |
| **1.4 Core code quality** | **5** | The pipeline is modular, well-structured and documented; appropriate choice and effective implementation of the selected toolkit. |
| **II. Task 2: DL Model & Data Pipeline** | **35** | Technical Depth and ML Proficiency |
| **2.1 Data Pipeline Quality** | **15** | Demonstrates clear steps for ingestion, cleaning, augmentation, and preparation. Code is well-structured and documented. |
| **2.2 Model Implementation** | **15** | Correct selection and implementation of the deep-learning model (classification or prediction). Appropriate choice of model and architecture for the selected dataset. |
| **2.3 Evaluation and Results** | **5** | Includes metrics (e.g., accuracy, F1-score, confusion matrix, R², MAE) to evaluate the model's performance. The model is trained and ready for deployment/calling. |
| **III. Task 3: Assistant Feature (Application Integration)** | **30** | Integration and Engineering |
| **3.1 Application Logic** | **15** | The application correctly identifies the requested task from the user's input (calibrate / filter / detect features / classify / measure). |
| **3.2 Tool Integration** | **15** | Seamless execution of the defined modules (running the Task 1 pipeline and the Task 2 model) and passing the visualized result back to the user (Requirements b & c). |
| **IV. Documentation and Submission** | **15** | Professionalism and Clarity |
| **4.1 Project Report** | **10** | Clear documentation of the dataset, model choice, pipeline steps, application workflow, and explanation of the implemented techniques. |
| **4.2 Code Quality \+ README.md** | **5** | Code is clean, modular, and utilizes appropriate variables/function names. (A brief demonstration video/GIF may be required for live testing). |

