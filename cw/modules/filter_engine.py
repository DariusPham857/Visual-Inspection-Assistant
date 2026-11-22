"""
Task 1: Spatial Filtering, Gradients, and Frequency Analysis Engine
Fulfills Sections 2.3 - 2.5 / Criterion 1.2 of FCVG101 Coursework
"""
import cv2
import numpy as np

def apply_spatial_filters(gray_img, gaussian_ksize=3, gaussian_sigma=0, median_ksize=3):
    # Ensure kernel sizes are odd
    if gaussian_ksize % 2 == 0: gaussian_ksize += 1
    if median_ksize % 2 == 0: median_ksize += 1
    
    gauss = cv2.GaussianBlur(gray_img, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    median = cv2.medianBlur(gray_img, median_ksize)
    return gauss, median

def apply_gradients_and_edges(gray_img, canny_low=50, canny_high=150):
    smooth = cv2.GaussianBlur(gray_img, (3, 3), 0)
    sobel_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalize magnitude for display
    mag_disp = np.clip(magnitude, 0, 255).astype(np.uint8)
    edges = cv2.Canny(smooth, canny_low, canny_high)
    return sobel_x, sobel_y, mag_disp, edges

def apply_fft_filtering(gray_img, cutoff_radius=0.05):
    # 2D FFT and shift
    fft = np.fft.fft2(gray_img)
    fft_shifted = np.fft.fftshift(fft)
    spectrum = np.log1p(np.abs(fft_shifted))
    
    # Masks
    h, w = gray_img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    
    sigma_f = max(cutoff_radius, 0.005)
    low_mask = np.exp(-(FX ** 2 + FY ** 2) / (2 * sigma_f ** 2))
    high_mask = 1.0 - low_mask
    
    # Filter and inverse FFT
    low_fft = fft_shifted * low_mask
    high_fft = fft_shifted * high_mask
    
    low_img = np.fft.ifft2(np.fft.ifftshift(low_fft)).real
    high_img = np.fft.ifft2(np.fft.ifftshift(high_fft)).real
    
    low_disp = np.clip(low_img, 0, 255).astype(np.uint8)
    high_disp = np.clip(np.abs(high_img), 0, 255).astype(np.uint8)
    reconstructed = np.clip(low_img + high_img, 0, 255).astype(np.uint8)
    
    mse = float(np.mean((gray_img.astype(np.float64) - (low_img + high_img)) ** 2))
    return spectrum, low_mask, high_mask, low_disp, high_disp, reconstructed, mse
