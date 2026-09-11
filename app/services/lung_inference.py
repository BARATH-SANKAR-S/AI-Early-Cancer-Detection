"""
Lung Cancer Inference Service
Uses EfficientNet-B4 with transfer learning for lung cancer detection.
Supports CT scan slices and chest X-ray images.
Includes Grad-CAM explainability heatmaps.

Classes:
    0 - normal       : No cancer detected
    1 - benign       : Benign nodule (non-cancerous)
    2 - malignant    : Malignant nodule (cancerous)
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


# ─────────────────────────────────────────────
# Class definitions
# ─────────────────────────────────────────────
LUNG_CLASSES = ['normal', 'benign', 'malignant']

LUNG_CLASS_LABELS = {
    'normal':    'Normal — No Cancer Detected',
    'benign':    'Benign Nodule — Non-cancerous',
    'malignant': 'Malignant Nodule — Cancer Detected',
}

LUNG_RISK_LEVELS = {
    'normal':    'low',
    'benign':    'medium',
    'malignant': 'high',
}


# ─────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet.
    Produces heatmap showing which image regions influenced the prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap for given input."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def overlay_heatmap_on_image(original_image_path, cam, output_size=(224, 224)):
    """
    Overlay Grad-CAM heatmap on original image.
    Returns base64-encoded PNG string for embedding in HTML.
    """
    try:
        # Load original image
        orig = Image.open(original_image_path).convert('RGB').resize(output_size)
        orig_np = np.array(orig, dtype=np.float32)

        # Resize CAM to match image using PIL
        cam_norm = np.clip(cam, 0, 1)
        cam_pil = Image.fromarray(np.uint8(255 * cam_norm)).resize(output_size, Image.BILINEAR)
        cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0

        # Apply JET colormap via matplotlib
        try:
            import matplotlib.colormaps as cmaps
            cmap = cmaps['jet']
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')

        heatmap_rgb = np.uint8(255 * cmap(cam_resized)[:, :, :3])

        # Blend
        overlay = np.uint8(np.clip(0.55 * orig_np + 0.45 * heatmap_rgb, 0, 255))
        overlay_img = Image.fromarray(overlay)

        buf = io.BytesIO()
        overlay_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        logger.warning(f"Heatmap overlay failed: {e}")
        return None


# ─────────────────────────────────────────────
# Inference class
# ─────────────────────────────────────────────
class LungCancerInference:
    """
    Lung cancer detection using EfficientNet-B4 with transfer learning.
    Detects normal, benign nodule, and malignant nodule classes.
    """

    def __init__(self):
        self.model = None
        self.grad_cam = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "EfficientNet-B4 (Lung Cancer)"
        self.num_classes = len(LUNG_CLASSES)
        self._load_model()

    def _build_model(self):
        """Build EfficientNet-B4 architecture with custom classifier head."""
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        # Replace classifier for 3-class lung cancer detection
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes),
        )
        return model

    def _load_model(self):
        """Load EfficientNet-B4 model. Uses fine-tuned weights if available,
        otherwise falls back to ImageNet pre-trained weights."""
        try:
            logger.info(f"Loading lung cancer model: {self.model_name}")
            self.model = self._build_model()

            # Try loading fine-tuned weights
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models',
                'lung_cancer_efficientnet.pth'
            )
            if os.path.exists(model_path):
                logger.info(f"Loading fine-tuned weights from: {model_path}")
                state = torch.load(model_path, map_location=self.device)
                # Support both raw state_dict and checkpoint dicts
                if isinstance(state, dict) and 'model_state_dict' in state:
                    self.model.load_state_dict(state['model_state_dict'])
                else:
                    self.model.load_state_dict(state)
                logger.info("Fine-tuned lung weights loaded successfully.")
            else:
                logger.warning(
                    f"Fine-tuned weights not found at {model_path}. "
                    "Using ImageNet pre-trained EfficientNet-B4 as base."
                )

            self.model.to(self.device)
            self.model.eval()

            # Register Grad-CAM on the last feature conv block
            target_layer = self.model.features[-1]
            self.grad_cam = GradCAM(self.model, target_layer)

            logger.info("Lung cancer model ready.")

        except Exception as e:
            logger.error(f"Error loading lung cancer model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    # ─── Image transforms ───────────────────────────────────────────
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    def _preprocess(self, image_path):
        """Load and preprocess image for EfficientNet input."""
        img = Image.open(image_path).convert('RGB')
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return tensor

    # ─── Prediction ─────────────────────────────────────────────────
    def predict(self, image_path, generate_heatmap=True):
        """
        Predict lung cancer class and confidence from image.

        Args:
            image_path (str): Absolute path to CT/X-ray image.
            generate_heatmap (bool): Whether to generate Grad-CAM overlay.

        Returns:
            dict: {
                predicted_class, label, confidence, risk_level,
                model_name, image_type, class_probabilities,
                heatmap_b64 (optional)
            }
        """
        if self.model is None:
            return {
                'error': 'Lung cancer model not loaded',
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'lung',
            }

        try:
            img_tensor = self._preprocess(image_path)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, dim=1)

            predicted_class = LUNG_CLASSES[predicted_idx.item()]
            confidence_score = confidence.item()

            result = {
                'predicted_class': predicted_class,
                'label': LUNG_CLASS_LABELS[predicted_class],
                'confidence': confidence_score,
                'risk_level': LUNG_RISK_LEVELS[predicted_class],
                'model_name': self.model_name,
                'image_type': 'lung',
                'cancer_type': 'lung_cancer',
                'class_probabilities': {
                    LUNG_CLASSES[i]: probs[0][i].item()
                    for i in range(self.num_classes)
                },
            }

            # Grad-CAM heatmap
            if generate_heatmap and self.grad_cam is not None:
                try:
                    img_tensor_grad = self._preprocess(image_path)
                    img_tensor_grad.requires_grad_(True)
                    cam = self.grad_cam.generate(
                        img_tensor_grad, class_idx=predicted_idx.item()
                    )
                    result['heatmap_b64'] = overlay_heatmap_on_image(
                        image_path, cam
                    )
                except Exception as he:
                    logger.warning(f"Grad-CAM failed: {he}")
                    result['heatmap_b64'] = None

            return result

        except Exception as e:
            logger.error(f"Lung prediction error: {e}")
            return {
                'error': str(e),
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'lung',
            }


# ─────────────────────────────────────────────
# Singleton + convenience function
# ─────────────────────────────────────────────
_lung_service = None


def get_lung_service():
    """Return singleton LungCancerInference instance."""
    global _lung_service
    if _lung_service is None:
        _lung_service = LungCancerInference()
    return _lung_service


def predict_lung(image_path, generate_heatmap=True):
    """
    Convenience function for lung cancer prediction.

    Args:
        image_path (str): Path to CT or chest X-ray image.
        generate_heatmap (bool): Generate Grad-CAM heatmap overlay.

    Returns:
        dict: Prediction results.
    """
    return get_lung_service().predict(image_path, generate_heatmap)
