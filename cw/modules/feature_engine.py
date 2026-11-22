"""
Task 1: Feature Detection and Matching Engine (SIFT vs ORB)
Fulfills Section 2.6 / Criterion 1.3 of FCVG101 Coursework
"""
import time
import cv2
import numpy as np

def extract_features(img_gray, method="SIFT", nfeatures=500):
    start = time.perf_counter()
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=nfeatures, edgeThreshold=15, patchSize=31)
        norm_type = cv2.NORM_HAMMING
        
    kps, des = detector.detectAndCompute(img_gray, None)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return kps, des, norm_type, elapsed_ms

def match_features(des1, des2, norm_type=cv2.NORM_L2, ratio_thresh=0.75):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0, 0.0
        
    start = time.perf_counter()
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    valid_pairs = 0
    for pair in knn_matches:
        if len(pair) == 2:
            valid_pairs += 1
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                
    elapsed_ms = (time.perf_counter() - start) * 1000
    retention_rate = (len(good_matches) / valid_pairs * 100) if valid_pairs > 0 else 0.0
    return good_matches, valid_pairs, retention_rate, elapsed_ms

def draw_matches_view(img1, kp1, img2, kp2, good_matches, max_display=40):
    sorted_matches = sorted(good_matches, key=lambda x: x.distance)[:max_display]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, sorted_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis
