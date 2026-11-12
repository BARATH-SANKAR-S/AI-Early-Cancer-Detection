"""
Doctor Annotation & Active Learning Service
Enables doctors to annotate AI predictions and provide feedback for model improvement.
Supports active learning through doctor feedback integration.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnnotationType(Enum):
    """Types of doctor annotations"""
    CORRECTION = "correction"  # Doctor corrects AI prediction
    CONFIRMATION = "confirmation"  # Doctor confirms AI prediction
    UNCERTAIN = "uncertain"  # Doctor is uncertain about prediction
    REQUEST_RESCAN = "rescan"  # Doctor requests additional imaging
    CLINICAL_NOTE = "clinical_note"  # General clinical observation


class ConfidenceLevel(Enum):
    """Doctor confidence levels in their annotations"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DoctorAnnotation:
    """Doctor annotation data structure"""
    prediction_id: int
    doctor_id: int
    annotation_type: AnnotationType
    corrected_label: Optional[str]  # If correction
    confidence_level: ConfidenceLevel
    clinical_notes: str
    annotation_timestamp: datetime
    additional_data: Optional[Dict] = None


@dataclass
class ActiveLearningData:
    """Data for active learning model improvement"""
    prediction_id: int
    original_prediction: str
    doctor_annotation: DoctorAnnotation
    disagreement_score: float
    learning_value: float
    should_retrain: bool


class AnnotationService:
    """
    Service for managing doctor annotations and active learning.
    Enables continuous model improvement through expert feedback.
    """
    
    def __init__(self):
        """Initialize the annotation service"""
        self.annotation_history = []
        self.learning_queue = []
    
    def create_annotation(
        self,
        prediction_id: int,
        doctor_id: int,
        annotation_type: str,
        corrected_label: Optional[str] = None,
        confidence_level: str = 'medium',
        clinical_notes: str = '',
        additional_data: Optional[Dict] = None
    ) -> DoctorAnnotation:
        """
        Create a new doctor annotation.
        
        Args:
            prediction_id: ID of the prediction being annotated
            doctor_id: ID of the doctor making the annotation
            annotation_type: Type of annotation
            corrected_label: Corrected label if this is a correction
            confidence_level: Doctor's confidence in their annotation
            clinical_notes: Additional clinical notes
            additional_data: Optional additional metadata
            
        Returns:
            DoctorAnnotation object
        """
        try:
            annotation = DoctorAnnotation(
                prediction_id=prediction_id,
                doctor_id=doctor_id,
                annotation_type=AnnotationType(annotation_type),
                corrected_label=corrected_label,
                confidence_level=ConfidenceLevel(confidence_level),
                clinical_notes=clinical_notes,
                annotation_timestamp=datetime.utcnow(),
                additional_data=additional_data or {}
            )
            
            self.annotation_history.append(annotation)
            logger.info(f"Created annotation for prediction {prediction_id} by doctor {doctor_id}")
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error creating annotation: {e}")
            raise
    
    def calculate_disagreement_score(
        self,
        original_prediction: str,
        doctor_annotation: DoctorAnnotation
    ) -> float:
        """
        Calculate disagreement score between AI prediction and doctor annotation.
        
        Args:
            original_prediction: Original AI prediction
            doctor_annotation: Doctor's annotation
            
        Returns:
            Disagreement score between 0.0 (no disagreement) and 1.0 (complete disagreement)
        """
        disagreement = 0.0
        
        if doctor_annotation.annotation_type == AnnotationType.CORRECTION:
            # If doctor corrected the prediction, high disagreement
            if doctor_annotation.corrected_label and doctor_annotation.corrected_label != original_prediction:
                disagreement = 0.8
            else:
                disagreement = 0.3
        
        elif doctor_annotation.annotation_type == AnnotationType.CONFIRMATION:
            # Doctor confirmed the prediction, low disagreement
            disagreement = 0.1
        
        elif doctor_annotation.annotation_type == AnnotationType.UNCERTAIN:
            # Doctor is uncertain, moderate disagreement
            disagreement = 0.5
        
        elif doctor_annotation.annotation_type == AnnotationType.REQUEST_RESCAN:
            # Doctor requested rescan, high disagreement
            disagreement = 0.7
        
        # Adjust by doctor confidence
        confidence_multiplier = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.8,
            ConfidenceLevel.LOW: 0.5
        }
        
        disagreement *= confidence_multiplier.get(doctor_annotation.confidence_level, 0.8)
        
        return min(disagreement, 1.0)
    
    def calculate_learning_value(
        self,
        disagreement_score: float,
        prediction_confidence: float,
        doctor_confidence: ConfidenceLevel
    ) -> float:
        """
        Calculate the learning value of an annotation for model improvement.
        
        Args:
            disagreement_score: Disagreement between AI and doctor
            prediction_confidence: Original AI prediction confidence
            doctor_confidence: Doctor's confidence in their annotation
            
        Returns:
            Learning value between 0.0 (low value) and 1.0 (high value)
        """
        # High disagreement + high doctor confidence = high learning value
        learning_value = disagreement_score
        
        # Adjust by doctor confidence
        confidence_weight = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4
        }
        learning_value *= confidence_weight.get(doctor_confidence, 0.7)
        
        # High AI confidence with disagreement is especially valuable
        if prediction_confidence > 0.8 and disagreement_score > 0.5:
            learning_value *= 1.2
        
        return min(learning_value, 1.0)
    
    def should_retrain_model(self, learning_queue: List[ActiveLearningData]) -> bool:
        """
        Determine if model should be retrained based on accumulated annotations.
        
        Args:
            learning_queue: Queue of active learning data
            
        Returns:
            True if model should be retrained, False otherwise
        """
        if len(learning_queue) < 10:
            return False
        
        # Calculate average learning value
        avg_learning_value = sum(data.learning_value for data in learning_queue) / len(learning_queue)
        
        # Retrain if average learning value is high enough
        return avg_learning_value > 0.6
    
    def process_annotation_for_active_learning(
        self,
        prediction_id: int,
        original_prediction: str,
        prediction_confidence: float,
        doctor_annotation: DoctorAnnotation
    ) -> ActiveLearningData:
        """
        Process annotation for active learning pipeline.
        
        Args:
            prediction_id: ID of the prediction
            original_prediction: Original AI prediction
            prediction_confidence: Original AI confidence score
            doctor_annotation: Doctor's annotation
            
        Returns:
            ActiveLearningData object
        """
        disagreement_score = self.calculate_disagreement_score(
            original_prediction, doctor_annotation
        )
        
        learning_value = self.calculate_learning_value(
            disagreement_score, prediction_confidence, doctor_annotation.confidence_level
        )
        
        active_learning_data = ActiveLearningData(
            prediction_id=prediction_id,
            original_prediction=original_prediction,
            doctor_annotation=doctor_annotation,
            disagreement_score=disagreement_score,
            learning_value=learning_value,
            should_retrain=learning_value > 0.7  # High-value annotations trigger retraining
        )
        
        self.learning_queue.append(active_learning_data)
        
        return active_learning_data
    
    def get_annotation_statistics(self) -> Dict:
        """
        Get statistics about annotations collected.
        
        Returns:
            Dictionary with annotation statistics
        """
        if not self.annotation_history:
            return {
                'total_annotations': 0,
                'by_type': {},
                'by_confidence': {},
                'avg_disagreement': 0.0,
                'high_value_annotations': 0,
                'learning_queue_size': len(self.learning_queue) if hasattr(self, 'learning_queue') and self.learning_queue else 0
            }
        
        by_type = {}
        by_confidence = {}
        total_disagreement = 0.0
        high_value_count = 0
        
        for annotation in self.annotation_history:
            # Count by type
            annotation_type = annotation.annotation_type.value
            by_type[annotation_type] = by_type.get(annotation_type, 0) + 1
            
            # Count by confidence
            confidence = annotation.confidence_level.value
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
        
        # Calculate learning queue statistics
        if self.learning_queue:
            total_disagreement = sum(data.disagreement_score for data in self.learning_queue)
            high_value_count = sum(1 for data in self.learning_queue if data.learning_value > 0.7)
        
        return {
            'total_annotations': len(self.annotation_history),
            'by_type': by_type,
            'by_confidence': by_confidence,
            'avg_disagreement': total_disagreement / len(self.learning_queue) if self.learning_queue else 0.0,
            'high_value_annotations': high_value_count,
            'learning_queue_size': len(self.learning_queue)
        }
    
    def get_high_priority_annotations(self, threshold: float = 0.7) -> List[ActiveLearningData]:
        """
        Get high-priority annotations for model retraining.
        
        Args:
            threshold: Learning value threshold
            
        Returns:
            List of high-priority active learning data
        """
        return [data for data in self.learning_queue if data.learning_value >= threshold]
    
    def export_annotations_for_training(self) -> List[Dict]:
        """
        Export annotations in format suitable for model retraining.
        
        Returns:
            List of annotation dictionaries for training
        """
        training_data = []
        
        for data in self.learning_queue:
            if data.learning_value > 0.5:  # Only include valuable annotations
                training_data.append({
                    'prediction_id': data.prediction_id,
                    'original_prediction': data.original_prediction,
                    'corrected_label': data.doctor_annotation.corrected_label,
                    'annotation_type': data.doctor_annotation.annotation_type.value,
                    'doctor_confidence': data.doctor_annotation.confidence_level.value,
                    'disagreement_score': data.disagreement_score,
                    'learning_value': data.learning_value,
                    'clinical_notes': data.doctor_annotation.clinical_notes,
                    'annotation_timestamp': data.doctor_annotation.annotation_timestamp.isoformat()
                })
        
        return training_data
    
    def clear_learning_queue(self):
        """Clear the learning queue after model retraining"""
        self.learning_queue = []
        logger.info("Learning queue cleared after model retraining")


# Singleton instance
_annotation_service = None

def get_annotation_service() -> AnnotationService:
    """Get or create the annotation service singleton"""
    global _annotation_service
    if _annotation_service is None:
        _annotation_service = AnnotationService()
    return _annotation_service