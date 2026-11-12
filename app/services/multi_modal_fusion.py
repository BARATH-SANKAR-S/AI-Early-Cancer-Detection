"""
Multi-Modal Fusion Service
Combines predictions from different modalities (clinical data, medical images)
into a single weighted fusion score with visual breakdown.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of diagnostic modalities"""
    CLINICAL = "clinical"
    IMAGE = "image"
    LAB = "lab"
    GENETIC = "genetic"


@dataclass
class ModalityScore:
    """Individual modality prediction score"""
    modality: ModalityType
    confidence: float
    prediction: str
    weight: float
    metadata: Optional[Dict] = None


@dataclass
class FusionResult:
    """Combined multi-modal fusion result"""
    fusion_score: float
    overall_risk_level: str
    confidence_interval: Tuple[float, float]
    modality_contributions: Dict[str, float]
    dominant_modality: str
    clinical_insight: str
    visual_breakdown: Dict


class MultiModalFusionService:
    """
    Service for combining multi-modal predictions into a single fusion score.
    Uses weighted averaging based on modality reliability and clinical importance.
    """
    
    # Default weights for different modalities (can be customized)
    DEFAULT_WEIGHTS = {
        ModalityType.IMAGE: 0.45,      # Medical imaging (highest weight)
        ModalityType.CLINICAL: 0.30,   # Clinical data and symptoms
        ModalityType.LAB: 0.15,        # Laboratory results
        ModalityType.GENETIC: 0.10     # Genetic markers
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'low': 0.0,
        'medium': 0.35,
        'high': 0.70
    }
    
    def __init__(self, custom_weights: Optional[Dict[ModalityType, float]] = None):
        """
        Initialize fusion service with optional custom weights.
        
        Args:
            custom_weights: Optional custom weights for modalities
        """
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def calculate_fusion_score(
        self,
        modality_scores: List[ModalityScore],
        clinical_context: Optional[Dict] = None
    ) -> FusionResult:
        """
        Calculate multi-modal fusion score from individual modality predictions.
        
        Args:
            modality_scores: List of individual modality scores
            clinical_context: Optional clinical context for adjustments
            
        Returns:
            FusionResult with combined score and breakdown
        """
        if not modality_scores:
            raise ValueError("At least one modality score required")
        
        # Calculate weighted fusion score
        weighted_sum = 0.0
        total_weight = 0.0
        contributions = {}
        
        for score in modality_scores:
            weight = self.weights.get(score.modality, 0.25)
            weighted_sum += score.confidence * weight
            total_weight += weight
            contributions[score.modality.value] = score.confidence * weight
        
        # Normalize if total weight is not 1.0
        if total_weight > 0:
            fusion_score = weighted_sum / total_weight
        else:
            fusion_score = weighted_sum
        
        # Determine risk level
        risk_level = self._determine_risk_level(fusion_score)
        
        # Calculate confidence interval (Monte Carlo approximation)
        confidence_interval = self._calculate_confidence_interval(modality_scores)
        
        # Find dominant modality
        dominant_modality = max(contributions, key=contributions.get)
        
        # Generate clinical insight
        clinical_insight = self._generate_clinical_insight(
            modality_scores, fusion_score, risk_level, clinical_context
        )
        
        # Create visual breakdown
        visual_breakdown = self._create_visual_breakdown(
            modality_scores, contributions, fusion_score
        )
        
        return FusionResult(
            fusion_score=fusion_score,
            overall_risk_level=risk_level,
            confidence_interval=confidence_interval,
            modality_contributions=contributions,
            dominant_modality=dominant_modality,
            clinical_insight=clinical_insight,
            visual_breakdown=visual_breakdown
        )
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on fusion score"""
        if score >= self.RISK_THRESHOLDS['high']:
            return 'high'
        elif score >= self.RISK_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_interval(
        self,
        modality_scores: List[ModalityScore]
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using variance across modalities.
        Simulates Monte Carlo uncertainty estimation.
        """
        if len(modality_scores) < 2:
            return (0.0, 0.0)
        
        confidences = [score.confidence for score in modality_scores]
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        
        # 95% confidence interval (2 standard deviations)
        margin_of_error = 2 * std_dev / (len(confidences) ** 0.5)
        
        lower_bound = max(0.0, mean_confidence - margin_of_error)
        upper_bound = min(1.0, mean_confidence + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def _generate_clinical_insight(
        self,
        modality_scores: List[ModalityScore],
        fusion_score: float,
        risk_level: str,
        clinical_context: Optional[Dict]
    ) -> str:
        """Generate clinical insight based on fusion results"""
        insights = []
        
        # Analyze modality agreement
        if len(modality_scores) > 1:
            confidences = [score.confidence for score in modality_scores]
            agreement = max(confidences) - min(confidences)
            
            if agreement < 0.2:
                insights.append("Strong agreement across diagnostic modalities.")
            elif agreement < 0.4:
                insights.append("Moderate agreement across diagnostic modalities.")
            else:
                insights.append("Significant variance between diagnostic modalities.")
        
        # Risk level specific insights
        if risk_level == 'high':
            insights.append("High-risk finding requires immediate clinical attention.")
        elif risk_level == 'medium':
            insights.append("Medium-risk finding suggests follow-up evaluation.")
        else:
            insights.append("Low-risk finding with routine monitoring recommended.")
        
        # Clinical context adjustments
        if clinical_context:
            if clinical_context.get('family_history') == 'multiple':
                insights.append("Family history of cancer may elevate risk profile.")
            if clinical_context.get('age', 0) > 60:
                insights.append("Age factor may contribute to increased risk.")
        
        return " ".join(insights)
    
    def _create_visual_breakdown(
        self,
        modality_scores: List[ModalityScore],
        contributions: Dict[str, float],
        fusion_score: float
    ) -> Dict:
        """
        Create visual breakdown data for frontend rendering.
        Includes charts, progress bars, and contribution metrics.
        """
        breakdown = {
            'fusion_score': {
                'value': fusion_score,
                'percentage': round(fusion_score * 100, 1),
                'color': self._get_risk_color(fusion_score)
            },
            'modality_contributions': [],
            'risk_gauge': {
                'low': self.RISK_THRESHOLDS['low'],
                'medium': self.RISK_THRESHOLDS['medium'],
                'high': self.RISK_THRESHOLDS['high'],
                'current': fusion_score
            },
            'confidence_interval': {
                'lower': 0.0,
                'upper': 0.0,
                'width': 0.0
            }
        }
        
        # Add modality contributions
        for modality, contribution in contributions.items():
            percentage = round(contribution * 100, 1)
            breakdown['modality_contributions'].append({
                'modality': modality,
                'contribution': contribution,
                'percentage': percentage,
                'color': self._get_modality_color(modality)
            })
        
        # Sort by contribution
        breakdown['modality_contributions'].sort(
            key=lambda x: x['contribution'], reverse=True
        )
        
        return breakdown
    
    def _get_risk_color(self, score: float) -> str:
        """Get color based on risk level"""
        if score >= self.RISK_THRESHOLDS['high']:
            return '#dc2626'  # Red
        elif score >= self.RISK_THRESHOLDS['medium']:
            return '#f59e0b'  # Orange
        else:
            return '#10b981'  # Green
    
    def _get_modality_color(self, modality: str) -> str:
        """Get color for specific modality"""
        colors = {
            'image': '#3b82f6',      # Blue
            'clinical': '#8b5cf6',   # Purple
            'lab': '#ec4899',        # Pink
            'genetic': '#14b8a6'     # Teal
        }
        return colors.get(modality, '#6b7280')  # Gray default


def create_modality_score(
    modality_type: str,
    confidence: float,
    prediction: str,
    weight: Optional[float] = None,
    metadata: Optional[Dict] = None
) -> ModalityScore:
    """
    Factory function to create ModalityScore objects.
    
    Args:
        modality_type: Type of modality (clinical, image, lab, genetic)
        confidence: Confidence score (0.0 to 1.0)
        prediction: Prediction result
        weight: Optional custom weight
        metadata: Optional metadata
        
    Returns:
        ModalityScore object
    """
    try:
        modality = ModalityType(modality_type.lower())
    except ValueError:
        logger.warning(f"Unknown modality type: {modality_type}, defaulting to clinical")
        modality = ModalityType.CLINICAL
    
    return ModalityScore(
        modality=modality,
        confidence=confidence,
        prediction=prediction,
        weight=weight or 0.25,
        metadata=metadata or {}
    )


# Singleton instance for easy access
_fusion_service = None

def get_fusion_service() -> MultiModalFusionService:
    """Get or create the fusion service singleton"""
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = MultiModalFusionService()
    return _fusion_service