"""
Task 1: Camera Calibration & Image Undistortion Engine
Fulfills Section 2.2 / Criterion 1.1 of FCVG101 Coursework
"""
import json
import cv2
import numpy as np
from pathlib import Path

def load_saved_calibration(json_path="camera_calibration_params.json"):
    path = Path(json_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix_K"], dtype=np.float64)
    dist = np.array(data["distortion_coefficients_D"], dtype=np.float64)
    rms = data.get("overall_rms_reprojection_error_px", None)
    return {
        "K": K,
        "dist": dist,
        "rms": rms,
        "resolution": data.get("image_resolution", [640, 480]),
        "pattern": data.get("checkerboard_pattern", [9, 6])
    }

def undistort_image(img, K, dist, alpha=0):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha, newImgSize=(w, h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)
    return undistorted, new_K, roi

def run_checkerboard_calibration(image_paths, pattern_size=(9, 6), square_size=1.0):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = pattern_size
    
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    
    objpoints = []
    imgpoints = []
    valid_paths = []
    sample_vis = None
    img_shape = None
    
    for p in image_paths:
        if isinstance(p, (str, Path)):
            img = cv2.imread(str(p))
        else:
            img = p # in-memory BGR numpy array
            
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
            
        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows), 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            objpoints.append(objp)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)
            valid_paths.append(str(p) if isinstance(p, (str, Path)) else "in_memory")
            if sample_vis is None:
                sample_vis = cv2.drawChessboardCorners(img.copy(), (cols, rows), corners_sub, ret)
                
    if len(objpoints) < 3:
        return None, "Not enough valid checkerboard images (need at least 3 views with all corners detected)."
        
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    rmse_per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = np.sqrt(np.mean(np.sum((imgpoints[i] - proj) ** 2, axis=2)))
        rmse_per_view.append(float(err))
        
    return {
        "success": True,
        "overall_rms": float(rms),
        "K": K,
        "dist": dist,
        "detected_count": len(objpoints),
        "total_count": len(image_paths),
        "rmse_per_view": rmse_per_view,
        "sample_vis": sample_vis,
        "img_shape": img_shape
    }, None
