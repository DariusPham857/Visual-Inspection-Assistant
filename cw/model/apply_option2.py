import json

with open("dataset.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0 is the Markdown cell for Section 2.0
cell0_text = """# Section 2: Classical Computer Vision Pipeline
*(Fulfills Task 1: Core Image Analysis Engine - 20 Points)*

## 2.0 Image Formation and Camera Calibration
*(Coursework Criterion 1.1 & Requirement a - 5 Points)*

In a computer vision system (such as an automated visual inspection assistant), raw images captured by optical cameras suffer from perspective projection distortions and optical lens aberrations. Camera calibration recovers the camera's internal geometry (intrinsic parameters) and lens distortion coefficients, establishing a metric mapping between 3D world coordinates and 2D pixel coordinates.

### 2.0.1 Mathematical Formulation: Pinhole Model & Non-linear Distortion

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
   $$s \\tilde{\\mathbf{x}} = K [R \\mid \\mathbf{t}] \\tilde{\\mathbf{X}}_w \\quad (\\text{where scale factor } s = Z_c)$$"""

nb["cells"][0]["source"] = [l + "\n" for l in cell0_text.split("\n")]
if nb["cells"][0]["source"][-1] == "\n":
    nb["cells"][0]["source"][-1] = ""

# Update headers in code cells 1, 2, 3, 4
def update_first_line(cell_idx, new_first_line):
    lines = nb["cells"][cell_idx]["source"]
    if lines:
        lines[0] = new_first_line + "\n"

update_first_line(1, "# 2.0.2 Checkerboard Corner Detection and Sub-pixel Refinement")
update_first_line(2, "# 2.0.3 Camera Calibration Execution (Zhang's Method)")
update_first_line(3, "# 2.0.4 Image Undistortion and Geometric Collinearity Verification")
update_first_line(4, "# 2.0.5 Quantitative Reprojection Error Analysis")

scratch_path = r"C:\Users\KHANH NINH\.gemini\antigravity-cli\brain\11ae0d3e-2d08-4a16-b69f-3435ec4214b4\scratch\dataset_option2.ipynb"
with open(scratch_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook to Option 2 in scratch.")
