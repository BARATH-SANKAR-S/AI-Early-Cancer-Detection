"""
Brain Tumor Inference Service
Uses ResNet50 (ct_brain_tumor_continued.pth) for brain tumor detection.
Supports Brain MRI and CT scan images.
Includes Grad-CAM explainability heatmaps.

Classes:
    0 - normal  : No tumor detected
    1 - tumor   : Brain tumor detected
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import logging
logger = logging.getLogger(__name__)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# ── Class definitions ─────────────────────────────────────────────────────────
BRAIN_CLASSES = ['normal', 'tumor']

BRAIN_CLASS_LABELS = {
    'normal':      'Normal — No Tumor Detected',
    'tumor':       'Brain Tumor Detected',
    'notumor':     'Normal — No Tumor Detected',
    'glioma':      'Glioma — Malignant Brain Tumor',
    'meningioma':  'Meningioma — Benign Brain Tumor',
    'pituitary':   'Pituitary Adenoma — Benign Brain Tumor',
    'nodule':      'Brain Nodule — Suspicious Finding',
}

def get_brain_risk_level(predicted_class, confidence=0.0):
    """
    Calculate risk level ('low', 'medium', 'high') for brain cancer detection
    based on predicted class and model confidence.
    
    Clinical Stratification:
    - High: Malignant tumors (glioma, malignant) or high-confidence tumor finding (>= 75%).
    - Medium: Benign or intermediate neoplasms (meningioma, pituitary, nodule),
              moderate-confidence tumor (< 75%), or borderline normal scans (< 70%).
    - Low: Confident normal scans (>= 70%) with no tumor detected.
    """
    cls = (predicted_class or '').lower().strip()
    conf = float(confidence or 0.0)

    if cls in ('glioma', 'malignant'):
        return 'high'
    elif cls in ('meningioma', 'pituitary', 'benign', 'nodule'):
        return 'medium'
    elif cls == 'tumor':
        return 'high' if conf >= 0.75 else 'medium'
    elif cls in ('normal', 'notumor'):
        return 'low' if conf >= 0.70 else 'medium'
    return 'medium' if conf >= 0.60 else 'low'


BRAIN_RISK_LEVELS = {
    'normal':      'low',
    'tumor':       'high',
    'notumor':     'low',
    'glioma':      'high',
    'meningioma':  'medium',
    'pituitary':   'medium',
    'nodule':      'medium',
}


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAMBrain:
    """Gradient-weighted Class Activation Mapping for ResNet50."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def _overlay_heatmap(image_path, cam, size=(224, 224)):
    """Overlay Grad-CAM heatmap on original image, return base64 PNG."""
    try:
        orig = Image.open(image_path).convert('RGB').resize(size)
        orig_np = np.array(orig, dtype=np.float32)

        cam_pil = Image.fromarray(np.uint8(255 * np.clip(cam, 0, 1))).resize(size, Image.BILINEAR)
        cam_r = np.array(cam_pil, dtype=np.float32) / 255.0

        try:
            import matplotlib.colormaps as cmaps
            cmap = cmaps['jet']
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')

        heatmap_rgb = np.uint8(255 * cmap(cam_r)[:, :, :3])
        overlay = np.uint8(0.55 * orig_np + 0.45 * heatmap_rgb)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        logger.warning(f'Heatmap overlay failed: {e}')
        return None


# ── Model loading (singleton) ─────────────────────────────────────────────────
_brain_model = None
_brain_device = None

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'ct_brain_tumor_continued.pth'
)
IMG_SIZE = 224


def _load_brain_model():
    global _brain_model, _brain_device

    if _brain_model is not None:
        return _brain_model, _brain_device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build ResNet50 with binary head (matches training script)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),   # binary: normal / tumor
    )

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Handle both raw state_dict and checkpoint dicts
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        model.load_state_dict(state, strict=False)
        logger.info(f'Brain model loaded from {MODEL_PATH}')
    else:
        logger.warning(f'Brain model not found at {MODEL_PATH}, using untrained weights')

    model.to(device)
    model.eval()

    _brain_model = model
    _brain_device = device
    return model, device


# ── Inference ─────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_brain(image_path, generate_heatmap=True):
    """
    Run brain tumor inference on a medical image.

    Args:
        image_path      (str): Absolute path to the image.
        generate_heatmap(bool): Whether to produce Grad-CAM overlay.

    Returns:
        dict: {
            predicted_class, confidence, label, risk_level,
            class_probabilities, is_malignant,
            heatmap_b64 (optional), model_name, cancer_type, image_type
        }
    """
    try:
        model, device = _load_brain_model()

        img = Image.open(image_path).convert('RGB')
        tensor = _transform(img).unsqueeze(0).to(device)

        # Grad-CAM setup (layer4 is the last conv block in ResNet50)
        heatmap_b64 = None
        if generate_heatmap:
            grad_cam = GradCAMBrain(model, model.layer4[-1])

        with torch.set_grad_enabled(generate_heatmap):
            if generate_heatmap:
                output = model(tensor)
            else:
                with torch.no_grad():
                    output = model(tensor)

        probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = BRAIN_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Generate Grad-CAM
        if generate_heatmap:
            try:
                cam = grad_cam.generate(tensor, class_idx=pred_idx)
                heatmap_b64 = _overlay_heatmap(image_path, cam)
            except Exception as e:
                logger.warning(f'Grad-CAM failed: {e}')

        class_probabilities = {
            BRAIN_CLASSES[i]: float(probs[i]) for i in range(len(BRAIN_CLASSES))
        }

        risk_level = get_brain_risk_level(pred_class, confidence)
        label = BRAIN_CLASS_LABELS.get(pred_class, pred_class.capitalize())

        return {
            'predicted_class':    pred_class,
            'confidence':         confidence,
            'label':              label,
            'risk_level':         risk_level,
            'class_probabilities': class_probabilities,
            'is_malignant':       pred_class in ('tumor', 'glioma', 'malignant'),
            'heatmap_b64':        heatmap_b64,
            'model_name':         'ResNet50-BrainCT',
            'cancer_type':        'brain_tumor',
            'image_type':         'brain_ct',
        }

    except Exception as e:
        logger.error(f'Brain inference error: {e}')
        return {
            'predicted_class': 'error',
            'confidence':      0.0,
            'error':           str(e),
            'cancer_type':     'brain_tumor',
            'image_type':      'brain_ct',
        }
