"""
Skin Cancer Inference Service
Uses EfficientNet-B3 with transfer learning for skin lesion classification.
Based on ISIC 2019 Challenge dataset classes.
Includes Grad-CAM explainability heatmaps.

Classes (ISIC standard):
    0 - melanoma      : Melanoma (most dangerous skin cancer)
    1 - nevus         : Melanocytic Nevus (benign mole)
    2 - bcc           : Basal Cell Carcinoma
    3 - bkl           : Benign Keratosis (solar lentigo/seborrheic keratosis)
    4 - akiec         : Actinic Keratosis / Intraepithelial Carcinoma
    5 - df            : Dermatofibroma (benign)
    6 - vasc          : Vascular Lesion
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
SKIN_CLASSES = ['melanoma', 'nevus', 'bcc', 'bkl', 'akiec', 'df', 'vasc']

SKIN_CLASS_LABELS = {
    'melanoma': 'Melanoma — Malignant Skin Cancer',
    'nevus':    'Melanocytic Nevus — Benign Mole',
    'bcc':      'Basal Cell Carcinoma — Skin Cancer',
    'bkl':      'Benign Keratosis — Non-cancerous',
    'akiec':    'Actinic Keratosis — Pre-cancerous Lesion',
    'df':       'Dermatofibroma — Benign Skin Lesion',
    'vasc':     'Vascular Lesion — Benign Vascular Anomaly',
}

SKIN_RISK_LEVELS = {
    'melanoma': 'high',
    'nevus':    'low',
    'bcc':      'high',
    'bkl':      'low',
    'akiec':    'medium',
    'df':       'low',
    'vasc':     'low',
}

# Whether class is malignant/cancerous
SKIN_IS_MALIGNANT = {
    'melanoma': True,
    'nevus':    False,
    'bcc':      True,
    'bkl':      False,
    'akiec':    False,  # Pre-cancerous but not yet malignant
    'df':       False,
    'vasc':     False,
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

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)

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
        orig = Image.open(original_image_path).convert('RGB').resize(output_size)
        orig_np = np.array(orig, dtype=np.float32)

        cam_norm = np.clip(cam, 0, 1)
        cam_pil = Image.fromarray(np.uint8(255 * cam_norm)).resize(output_size, Image.BILINEAR)
        cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0

        import matplotlib
        cmap = matplotlib.colormaps['jet']
        heatmap_rgb = np.uint8(255 * cmap(cam_resized)[:, :, :3])

        # Raw heatmap
        buf_raw = io.BytesIO()
        Image.fromarray(heatmap_rgb).save(buf_raw, format='PNG')
        raw_b64 = f"data:image/png;base64,{base64.b64encode(buf_raw.getvalue()).decode('utf-8')}"

        # Blend overlay
        overlay = np.uint8(np.clip(0.55 * orig_np + 0.45 * heatmap_rgb, 0, 255))
        overlay_img = Image.fromarray(overlay)

        buf = io.BytesIO()
        overlay_img.save(buf, format='PNG')
        overlay_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        return overlay_b64, raw_b64

    except Exception as e:
        logger.warning(f"Heatmap overlay failed: {e}")
        return None, None


# ─────────────────────────────────────────────
# Inference class
# ─────────────────────────────────────────────
class SkinCancerInference:
    """
    Skin cancer detection using EfficientNet-B3 with transfer learning.
    Classifies dermoscopy images into 7 ISIC standard lesion categories.
    """

    def __init__(self):
        self.model = None
        self.grad_cam = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "EfficientNet-B3 (Skin Cancer ISIC)"
        self.num_classes = len(SKIN_CLASSES)
        self._load_model()

    def _build_model(self):
        """Build EfficientNet-B3 architecture with custom 7-class classifier."""
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
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
        """Load model. Uses fine-tuned ISIC weights if available."""
        try:
            logger.info(f"Loading skin cancer model: {self.model_name}")
            self.model = self._build_model()

            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models',
                'skin_cancer_efficientnet.pth'
            )
            if os.path.exists(model_path):
                logger.info(f"Loading fine-tuned weights from: {model_path}")
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    self.model.load_state_dict(state['model_state_dict'])
                else:
                    self.model.load_state_dict(state)
                logger.info("Fine-tuned skin weights loaded successfully.")
            else:
                logger.warning(
                    f"Fine-tuned weights not found at {model_path}. "
                    "Using ImageNet pre-trained EfficientNet-B3 as base."
                )

            self.model.to(self.device)
            self.model.eval()

            # Grad-CAM on last feature block
            target_layer = self.model.features[-1]
            self.grad_cam = GradCAM(self.model, target_layer)

            logger.info("Skin cancer model ready.")

        except Exception as e:
            logger.error(f"Error loading skin cancer model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    # ─── Image transforms (ISIC standard) ───────────────────────────
    _transform = transforms.Compose([
        transforms.Resize((300, 300)),           # EfficientNet-B3 native resolution
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    def _preprocess(self, image_path):
        """Load and preprocess dermoscopy image for EfficientNet-B3."""
        img = Image.open(image_path).convert('RGB')
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return tensor

    # ─── Prediction ─────────────────────────────────────────────────
    def predict(self, image_path, generate_heatmap=True):
        """
        Predict skin cancer class and confidence from image.

        Args:
            image_path (str): Absolute path to dermoscopy/skin image.
            generate_heatmap (bool): Whether to generate Grad-CAM overlay.

        Returns:
            dict: {
                predicted_class, label, confidence, risk_level,
                is_malignant, model_name, image_type,
                class_probabilities, heatmap_b64 (optional)
            }
        """
        if self.model is None:
            return {
                'error': 'Skin cancer model not loaded',
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'skin',
            }

        try:
            img_tensor = self._preprocess(image_path)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, dim=1)

            predicted_class = SKIN_CLASSES[predicted_idx.item()]
            confidence_score = confidence.item()

            result = {
                'predicted_class': predicted_class,
                'label': SKIN_CLASS_LABELS[predicted_class],
                'confidence': confidence_score,
                'risk_level': SKIN_RISK_LEVELS[predicted_class],
                'is_malignant': SKIN_IS_MALIGNANT[predicted_class],
                'model_name': self.model_name,
                'image_type': 'skin',
                'cancer_type': 'skin_cancer',
                'class_probabilities': {
                    SKIN_CLASSES[i]: probs[0][i].item()
                    for i in range(self.num_classes)
                },
            }

            # Grad-CAM heatmap
            result['heatmap_b64'] = None
            result['raw_heatmap_b64'] = None
            if generate_heatmap and self.grad_cam is not None:
                try:
                    img_tensor_grad = self._preprocess(image_path)
                    img_tensor_grad.requires_grad_(True)
                    cam = self.grad_cam.generate(
                        img_tensor_grad, class_idx=predicted_idx.item()
                    )
                    over_b64, raw_b64 = overlay_heatmap_on_image(
                        image_path, cam
                    )
                    result['heatmap_b64'] = over_b64
                    result['raw_heatmap_b64'] = raw_b64
                except Exception as he:
                    logger.warning(f"Grad-CAM failed: {he}")

            return result

        except Exception as e:
            logger.error(f"Skin prediction error: {e}")
            return {
                'error': str(e),
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'skin',
            }


# ─────────────────────────────────────────────
# Singleton + convenience function
# ─────────────────────────────────────────────
_skin_service = None


def get_skin_service():
    """Return singleton SkinCancerInference instance."""
    global _skin_service
    if _skin_service is None:
        _skin_service = SkinCancerInference()
    return _skin_service


def predict_skin(image_path, generate_heatmap=True):
    """
    Convenience function for skin cancer prediction.

    Args:
        image_path (str): Path to dermoscopy or skin photo image.
        generate_heatmap (bool): Generate Grad-CAM heatmap overlay.

    Returns:
        dict: Prediction results with ISIC class labels.
    """
    return get_skin_service().predict(image_path, generate_heatmap)
