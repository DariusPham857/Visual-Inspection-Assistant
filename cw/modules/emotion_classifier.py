"""
Task 2 & 3: Deep Learning Emotion Classification & Inference Engine
Includes YuNet Face Detector, Heuristic Margin Cropping, Preprocessing & CNN Classification
Fulfills Sections 3.3, 3.6, 4.2.2, 4.4.2 & 4.4.3 of FCVG101 Coursework
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Block 1: 1 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 24x24
            nn.Dropout2d(0.2)
        )
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 12x12
            nn.Dropout2d(0.25)
        )
        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 6x6
            nn.Dropout2d(0.3)
        )
        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 3x3
            nn.Dropout2d(0.4)
        )
        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


class YuNetFaceDetector:
    """
    OpenCV 5.0.0 Deep Learning Face Detector wrapping FaceDetectorYN.
    Maintains aspect ratio, limits max dimension to max_dim (640 px),
    and strictly maps bounding boxes back to original image coordinates.
    """
    def __init__(self, onnx_model_path="modules/face_detection_yunet_2023mar.onnx", score_threshold=0.50, nms_threshold=0.30):
        model_p = Path(onnx_model_path)
        if not model_p.exists():
            # Check fallback in parent or root
            alt_p = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
            if alt_p.exists():
                model_p = alt_p
            else:
                raise FileNotFoundError(f"YuNet ONNX model not found at {onnx_model_path}")
                
        self.model_path = str(model_p)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.detector = cv2.FaceDetectorYN_create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )

    def detect_faces(self, img_bgr, conf_thresh=None, max_dim=640):
        """
        Detects faces in img_bgr while preserving aspect ratio.
        Coordinates are scaled and clamped to [0, W_orig] and [0, H_orig].
        Returns list of dicts:
            [{"bbox": [x, y, w, h], "score": float, "landmarks": [...]}, ...]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h_orig, w_orig = img_bgr.shape[:2]
        if w_orig == 0 or h_orig == 0:
            return []

        # 1. Aspect-ratio preserving scaling
        scale = min(1.0, float(max_dim) / max(w_orig, h_orig))
        w_net = max(1, int(round(w_orig * scale)))
        h_net = max(1, int(round(h_orig * scale)))

        if scale < 1.0:
            img_net = cv2.resize(img_bgr, (w_net, h_net), interpolation=cv2.INTER_AREA)
        else:
            img_net = img_bgr

        # 2. Update dynamic network size & score threshold
        self.detector.setInputSize((w_net, h_net))
        if conf_thresh is not None:
            self.detector.setScoreThreshold(float(conf_thresh))
        else:
            self.detector.setScoreThreshold(self.score_threshold)

        # 3. Detect
        ret, faces = self.detector.detect(img_net)
        if faces is None or len(faces) == 0:
            return []

        # 4. Map back to original coordinate system
        detected = []
        for f in faces:
            box = f[:4] / scale
            bx, by, bw, bh = box
            x1 = int(np.clip(round(bx), 0, w_orig))
            y1 = int(np.clip(round(by), 0, h_orig))
            x2 = int(np.clip(round(bx + bw), 0, w_orig))
            y2 = int(np.clip(round(by + bh), 0, h_orig))
            cw = max(0, x2 - x1)
            ch = max(0, y2 - y1)
            
            if cw > 0 and ch > 0:
                score = float(f[14])
                landmarks = (f[4:14] / scale).tolist()
                detected.append({
                    "bbox": [x1, y1, cw, ch],
                    "score": score,
                    "landmarks": landmarks
                })
        return detected


def crop_face_with_margin(img_bgr, bbox, margin_ratio=0.10):
    """
    Expands bounding box by margin_ratio in width and height,
    then uses np.clip to clamp strictly within image boundaries.
    """
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    
    dx = int(round(margin_ratio * w))
    dy = int(round(margin_ratio * h))
    
    x1 = int(np.clip(x - dx, 0, w_img))
    y1 = int(np.clip(y - dy, 0, h_img))
    x2 = int(np.clip(x + w + dx, 0, w_img))
    y2 = int(np.clip(y + h + dy, 0, h_img))
    
    crop = img_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2 - x1, y2 - y1)


def preprocess_face_to_tensor(face_img, train_mean=0.507739, train_std=0.255078, device="cpu"):
    """
    Converts face crop to grayscale, resizes to (48, 48) using cv2.INTER_AREA,
    standardizes with training set statistics: (I/255.0 - mean) / std.
    Returns:
        tensor: torch.Tensor of shape (1, 1, 48, 48), dtype=torch.float32 on device
        preview_48: np.ndarray uint8 of shape (48, 48)
    """
    if face_img is None or face_img.size == 0:
        raise ValueError("Invalid face image crop.")
        
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
        
    if gray.shape != (48, 48):
        preview_48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    else:
        preview_48 = gray.copy()
        
    norm = (preview_48.astype(np.float32) / 255.0 - train_mean) / train_std
    tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, preview_48


class EmotionClassifier:
    """
    Integrated Face Detection and Facial Affect Classification Engine.
    Combines YuNet Face Detector and trained EmotionCNN checkpoint.
    """
    def __init__(self, checkpoint_path="model/fer2013_checkpoint.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        chk_path = Path(checkpoint_path)
        if not chk_path.exists():
            fallbacks = [
                Path(r"C:\Users\KHANH NINH\.gemini\antigravity-cli\brain\11ae0d3e-2d08-4a16-b69f-3435ec4214b4\scratch\fer2013_checkpoint.pth"),
                Path("scratch/fer2013_checkpoint.pth"),
                Path("model/fer2013_checkpoint.pth"),
                Path("best_fer_model.pth")
            ]
            for fb in fallbacks:
                if fb.exists():
                    chk_path = fb
                    break
            else:
                raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        checkpoint = torch.load(chk_path, map_location=self.device, weights_only=False)
        
        self.class_names = checkpoint.get("class_names", checkpoint.get("classes", CLASSES))
        self.train_mean = float(checkpoint.get("train_mean", checkpoint.get("mean", 0.507739)))
        self.train_std = float(checkpoint.get("train_std", checkpoint.get("std", 0.255078)))
        self.input_size = checkpoint.get("input_size", (1, 48, 48))
        self.best_val_loss = checkpoint.get("best_val_loss", None)
        
        self.model = EmotionCNN(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval() # Freeze BatchNorm and disable Dropout for inference
        
        # Initialize face detector
        try:
            self.face_detector = YuNetFaceDetector()
        except Exception as e:
            self.face_detector = None
            print(f"Warning: YuNet face detector could not be initialized: {e}")

    @torch.no_grad()
    def predict_tensor(self, tensor):
        """
        Evaluates preprocessed (1, 1, 48, 48) tensor.
        Returns pred_label (str), prob_dict (dict), top1_score (float).
        """
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        top1_score = float(probs[pred_idx])
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return pred_label, prob_dict, top1_score

    def predict_crop(self, face_crop):
        """
        Direct inference for already cropped face (e.g. bypass detector mode).
        Returns pred_label, prob_dict, preview_48, top1_score.
        """
        tensor, preview_48 = preprocess_face_to_tensor(
            face_crop, train_mean=self.train_mean, train_std=self.train_std, device=self.device
        )
        pred_label, prob_dict, top1_score = self.predict_tensor(tensor)
        return pred_label, prob_dict, preview_48, top1_score

    def detect_and_classify(self, img_bgr, conf_thresh=0.50, margin_ratio=0.10):
        """
        Runs YuNet detector, crops each detected face with margin_ratio,
        evaluates EmotionCNN on each face, and draws annotated bounding boxes.
        Returns:
            face_results: list of dicts with keys:
                - face_idx: 1-based int
                - bbox: [x, y, w, h]
                - score: float (detector confidence)
                - pred_label: str
                - top1_score: float
                - prob_dict: dict
                - crop_preview: np.ndarray (48x48)
            annotated_img: copy of img_bgr with drawn boxes and labels
        """
        if self.face_detector is None:
            raise RuntimeError("YuNet face detector is not initialized.")
            
        faces = self.face_detector.detect_faces(img_bgr, conf_thresh=conf_thresh)
        annotated_img = img_bgr.copy()
        face_results = []
        
        for idx, f in enumerate(faces):
            bbox = f["bbox"]
            crop, padded_box = crop_face_with_margin(img_bgr, bbox, margin_ratio=margin_ratio)
            
            if crop.size > 0:
                pred_label, prob_dict, preview_48, top1_score = self.predict_crop(crop)
                
                # Draw on annotated image
                x, y, w, h = bbox
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label_txt = f"#{idx+1}: {pred_label.capitalize()} ({top1_score*100:.1f}%)"
                # Text background
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x, max(0, y - th - 6)), (x + tw + 4, y), (0, 255, 0), -1)
                cv2.putText(annotated_img, label_txt, (x + 2, max(th, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            
                face_results.append({
                    "face_idx": idx + 1,
                    "bbox": bbox,
                    "padded_bbox": padded_box,
                    "det_score": f["score"],
                    "pred_label": pred_label,
                    "top1_score": top1_score,
                    "prob_dict": prob_dict,
                    "crop_preview": preview_48,
                    "crop_bgr": crop
                })
                
        return face_results, annotated_img

    # Legacy compatibility method
    def predict(self, image_input):
        """
        Backward compatible inference on a single image.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                img = image_input
        else:
            raise ValueError("Unsupported image input type.")
            
        pred_label, prob_dict, preview_48, _ = self.predict_crop(img)
        return pred_label, prob_dict, preview_48
