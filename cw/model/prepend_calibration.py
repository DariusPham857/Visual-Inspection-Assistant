import json

# Read existing notebook
with open("dataset.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

existing_cells = nb["cells"]

def make_cell(cell_type, source_text):
    lines = [line + "\n" for line in source_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

section_1_1_cells = [
    make_cell("markdown", """# Task 1: Core Image Analysis Engine

## 1.1 Image Formation and Camera Calibration

In a computer vision system (such as an automated visual inspection assistant), raw images captured by optical cameras suffer from perspective projection distortions and optical lens aberrations. Camera calibration recovers the camera's internal geometry (intrinsic parameters) and lens distortion coefficients, establishing a metric mapping between 3D world coordinates and 2D pixel coordinates.

### 1.1.1 Mathematical Formulation: Pinhole Model & Non-linear Distortion

1. **Homogeneous Pinhole Camera Model**:
   Given a 3D world point in homogeneous coordinates $\\tilde{\\mathbf{X}}_w = [X_w, Y_w, Z_w, 1]^T$, the extrinsic rigid-body transformation transforms it into the camera coordinate system $\\mathbf{X}_c = [X_c, Y_c, Z_c]^T$:
   $$\\mathbf{X}_c = [R \\mid \\mathbf{t}] \\tilde{\\mathbf{X}}_w$$
   Normalized pinhole projection onto the camera plane yields continuous coordinates:
   $$(x', y') = \\left(\\frac{X_c}{Z_c}, \\frac{Y_c}{Z_c}\\right), \\quad r^2 = x'^2 + y'^2$$

2. **Optical Lens Distortion Model**:
   Real-world lenses introduce non-linear spatial displacement in the normalized image plane before sensor digitization:
   - **Radial distortion** (curvature along concentric circles):
     $$x_{\\text{radial}} = x'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6), \\quad y_{\\text{radial}} = y'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
   - **Tangential distortion** (decentering/non-parallel lens elements):
     $$x_{\\text{tangential}} = 2 p_1 x' y' + p_2 (r^2 + 2 x'^2), \\quad y_{\\text{tangential}} = p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'$$
   Combining both yields distorted continuous coordinates $(x'', y'')$:
   $$x'' = x_{\\text{radial}} + x_{\\text{tangential}}, \\quad y'' = y_{\\text{radial}} + y_{\\text{tangential}}$$

3. **Sensor Digitization via Intrinsic Matrix $K$**:
   Finally, pixel coordinates $\\tilde{\\mathbf{x}} = [u, v, 1]^T$ are digitized on the camera sensor:
   $$\\tilde{\\mathbf{x}} = K \\begin{bmatrix} x'' \\\\ y'' \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} x'' \\\\ y'' \\\\ 1 \\end{bmatrix}$$
   In the ideal linear case (zero lens distortion, $x'' = x', y'' = y'$), the full projection simplifies to:
   $$s \\tilde{\\mathbf{x}} = K [R \\mid \\mathbf{t}] \\tilde{\\mathbf{X}}_w \\quad (\\text{where scale factor } s = Z_c)$$"""),

    make_cell("code", """# 1.1.2 Checkerboard Corner Detection and Sub-pixel Refinement
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

# Setup directory for calibration sequence
calib_dir = Path("calibration_images")
calib_dir.mkdir(exist_ok=True)

# Standard OpenCV calibration set: left01.jpg to left14.jpg (excluding absent left10)
base_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
img_names = [f"left{i:02d}.jpg" for i in range(1, 15) if i != 10]

for name in img_names:
    p = calib_dir / name
    if not p.exists():
        try:
            urllib.request.urlretrieve(base_url + name, p)
        except Exception as e:
            print(f"Note: Could not retrieve {name}: {e}")

# Verified checkerboard parameters for this dataset: 9 columns x 6 rows of internal corners
CHECKERBOARD = (9, 6)
# Square size is defined as 1.0 (grid units) as physical metric dimensions are unmeasured
SQUARE_SIZE = 1.0 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Generate 3D planar object points (Z = 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] # 3D real-world coordinates
imgpoints = [] # 2D refined sub-pixel coordinates
valid_image_paths = []

image_files = sorted(list(calib_dir.glob("*.jpg")))
sample_vis_img = None
img_shape = None

for fname in image_files:
    img = cv2.imread(str(fname))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = gray.shape[::-1]
    
    # Detect internal chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    # Filter strictly: only views with all 54 inner corners detected enter calibration
    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)
        valid_image_paths.append(fname)
        
        if sample_vis_img is None:
            sample_vis_img = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_refined, ret)

print(f"Total calibration views inspected : {len(image_files)}")
print(f"Successful corner detections      : {len(objpoints)} / {len(image_files)} ({len(objpoints)/len(image_files)*100:.1f}%)")

if sample_vis_img is not None:
    plt.figure(figsize=(9, 6))
    plt.imshow(cv2.cvtColor(sample_vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sub-pixel Checkerboard Corners: {valid_image_paths[0].name} ({CHECKERBOARD[0]}x{CHECKERBOARD[1]} corners)")
    plt.axis("off")
    plt.show()"""),

    make_cell("code", """# 1.1.3 Camera Calibration Execution (Zhang's Method)
overall_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("=" * 60)
print("CAMERA CALIBRATION PARAMETERS")
print("=" * 60)
print(f"Overall RMS Reprojection Error: {overall_rms:.4f} pixels\\n")

print("Intrinsic Camera Matrix K (3x3):")
print(np.array2string(K, precision=4, suppress_small=True))
print(f"  Focal lengths   : fx = {K[0,0]:.2f} px, fy = {K[1,1]:.2f} px")
print(f"  Principal point : cx = {K[0,2]:.2f} px, cy = {K[1,2]:.2f} px")
print(f"  Aspect ratio (fx / fy): {K[0,0]/K[1,1]:.4f}\\n")

# Dynamic inspection of distortion coefficients
dist_flat = dist.ravel()
param_names = ["k1 (radial)", "k2 (radial)", "p1 (tangential)", "p2 (tangential)", "k3 (radial)", 
               "k4 (radial)", "k5 (radial)", "k6 (radial)"]

print(f"Distortion Coefficients Vector D ({len(dist_flat)} parameters):")
for i, val in enumerate(dist_flat):
    lbl = param_names[i] if i < len(param_names) else f"coeff_{i}"
    print(f"  {lbl:18s}: {val:+.6e}")"""),

    make_cell("code", """# 1.1.4 Image Undistortion and Geometric Collinearity Verification
test_img = cv2.imread(str(valid_image_paths[0]))
h, w = test_img.shape[:2]

# Optimal new camera matrix (alpha = 0 eliminates invalid black borders)
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0, newImgSize=(w, h))
undistorted_img = cv2.undistort(test_img, K, dist, None, new_K)

# Quantitative Proof: Check collinearity of an outer row of corners
# In a distorted image, corners on an edge curve away from the straight chord line.
# In an undistorted image, they fall onto a straight line.
def measure_row_collinearity(corners_2d, cols=9, row_idx=0):
    row_pts = corners_2d[row_idx * cols : (row_idx + 1) * cols].reshape(-1, 2)
    p_start, p_end = row_pts[0], row_pts[-1]
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    # Distance from each intermediate point to the chord line
    distances = [
        abs(line_vec[0] * (p_start[1] - p[1]) - line_vec[1] * (p_start[0] - p[0])) / line_len
        for p in row_pts
    ]
    return row_pts, max(distances)

# Re-detect corners on undistorted image
gray_undist = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
ret_u, corners_u = cv2.findChessboardCorners(gray_undist, CHECKERBOARD, None)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Distorted Image\\n(Barrel optical distortion)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Undistorted Rectified Image\\n(Rectilinear geometry restored)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

if ret_u:
    c_u = cv2.cornerSubPix(gray_undist, corners_u, (11, 11), (-1, -1), criteria)
    pts_orig, max_dev_orig = measure_row_collinearity(imgpoints[0], cols=CHECKERBOARD[0], row_idx=0)
    pts_undist, max_dev_undist = measure_row_collinearity(c_u, cols=CHECKERBOARD[0], row_idx=0)
    print(f"Outer Row Collinearity Maximum Deviation:")
    print(f"  Distorted View   : {max_dev_orig:.3f} px deviation from chord line")
    print(f"  Undistorted View : {max_dev_undist:.3f} px deviation from chord line")
    print(f"  Curvature Error Reduction: {(1.0 - max_dev_undist / max_dev_orig) * 100:.1f}%")"""),

    make_cell("code", """# 1.1.5 Quantitative Reprojection Error Analysis
rmse_per_view = []

for i in range(len(objpoints)):
    projected_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    sq_err = np.sum((imgpoints[i] - projected_pts) ** 2, axis=2)
    view_rmse = np.sqrt(np.mean(sq_err))
    rmse_per_view.append(view_rmse)

plt.figure(figsize=(10, 4))
view_labels = [p.stem for p in valid_image_paths]
plt.bar(range(len(rmse_per_view)), rmse_per_view, color="#1f77b4", edgecolor="black", width=0.6)
plt.axhline(overall_rms, color="red", linestyle="--", linewidth=1.5,
            label=f"Overall RMS (Global Benchmark): {overall_rms:.4f} px")

plt.xticks(range(len(rmse_per_view)), view_labels, rotation=45)
plt.xlabel("Calibration View")
plt.ylabel("RMSE (pixels)")
plt.title("Reprojection RMSE per View")
plt.legend()
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

print(f"Overall RMS Reprojection Error (from calibrateCamera) : {overall_rms:.4f} px")
print(f"Mean of per-view RMSEs                                  : {np.mean(rmse_per_view):.4f} px")
print(f"Min RMSE per view                                       : {np.min(rmse_per_view):.4f} px ({view_labels[np.argmin(rmse_per_view)]})")
print(f"Max RMSE per view                                       : {np.max(rmse_per_view):.4f} px ({view_labels[np.argmax(rmse_per_view)]})")""")
]

# Prepend section 1.1 to the top of notebook
nb["cells"] = section_1_1_cells + existing_cells

scratch_output = r"C:\Users\KHANH NINH\.gemini\antigravity-cli\brain\11ae0d3e-2d08-4a16-b69f-3435ec4214b4\scratch\dataset_with_calib.ipynb"
with open(scratch_output, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated new notebook at {scratch_output}. Total cells: {len(nb['cells'])}")
