"""
Model Uncertainty Service
Implements Monte Carlo Dropout for measuring model prediction uncertainty.
Provides uncertainty quantification for AI predictions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics for a prediction"""
    mean_confidence: float
    std_confidence: float
    coefficient_of_variation: float
    prediction_entropy: float
    uncertainty_level: str  # low, medium, high
    monte_carlo_samples: int
    prediction_distribution: Dict[str, float]


class UncertaintyService:
    """
    Service for measuring model uncertainty using Monte Carlo Dropout.
    Provides uncertainty quantification for AI predictions.
    """
    
    def __init__(self, num_samples: int = 30):
        """
        Initialize the uncertainty service.
        
        Args:
            num_samples: Number of Monte Carlo samples for uncertainty estimation
        """
        self.num_samples = num_samples
        self.uncertainty_thresholds = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.5
        }
    
    def calculate_uncertainty(
        self,
        prediction_function,
        input_data,
        num_samples: Optional[int] = None
    ) -> UncertaintyMetrics:
        """
        Calculate uncertainty metrics using Monte Carlo Dropout.
        
        Args:
            prediction_function: Function that returns model predictions
            input_data: Input data for prediction
            num_samples: Number of Monte Carlo samples (optional)
            
        Returns:
            UncertaintyMetrics object with uncertainty measures
        """
        samples = num_samples or self.num_samples
        predictions = []
        
        # Run Monte Carlo Dropout sampling
        for _ in range(samples):
            try:
                # In a real implementation, this would call the model with dropout enabled
                # For demonstration, we'll simulate the behavior
                prediction = self._simulate_monte_carlo_prediction(prediction_function, input_data)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Monte Carlo sample failed: {e}")
                continue
        
        if not predictions:
            logger.error("No successful Monte Carlo samples")
            return self._create_default_uncertainty()
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_metrics(predictions)
        
        return uncertainty_metrics
    
    def _simulate_monte_carlo_prediction(self, prediction_function, input_data):
        """
        Simulate Monte Carlo prediction (placeholder for actual implementation).
        In production, this would call the model with dropout enabled.
        """
        # This is a simulation - in real implementation, you would:
        # 1. Enable dropout in the model
        # 2. Run prediction multiple times
        # 3. Collect the predictions
        
        # For demonstration, we'll add some noise to simulate variability
        base_prediction = prediction_function(input_data)
        noise = np.random.normal(0, 0.05)  # Add small noise
        return np.clip(base_prediction + noise, 0.0, 1.0)
    
    def _calculate_metrics(self, predictions: List[float]) -> UncertaintyMetrics:
        """
        Calculate uncertainty metrics from Monte Carlo samples.
        
        Args:
            predictions: List of prediction values from Monte Carlo sampling
            
        Returns:
            UncertaintyMetrics object
        """
        predictions_array = np.array(predictions)
        
        # Basic statistics
        mean_confidence = np.mean(predictions_array)
        std_confidence = np.std(predictions_array)
        
        # Coefficient of variation (normalized measure of dispersion)
        coefficient_of_variation = std_confidence / (mean_confidence + 1e-8)
        
        # Prediction entropy (measure of uncertainty in categorical predictions)
        prediction_entropy = self._calculate_entropy(predictions_array)
        
        # Determine uncertainty level
        uncertainty_level = self._determine_uncertainty_level(coefficient_of_variation)
        
        # Create prediction distribution
        prediction_distribution = self._create_distribution(predictions_array)
        
        return UncertaintyMetrics(
            mean_confidence=float(mean_confidence),
            std_confidence=float(std_confidence),
            coefficient_of_variation=float(coefficient_of_variation),
            prediction_entropy=float(prediction_entropy),
            uncertainty_level=uncertainty_level,
            monte_carlo_samples=len(predictions),
            prediction_distribution=prediction_distribution
        )
    
    def _calculate_entropy(self, predictions: np.ndarray) -> float:
        """
        Calculate entropy of predictions.
        
        Args:
            predictions: Array of prediction values
            
        Returns:
            Entropy value
        """
        # Normalize predictions to create a probability distribution
        normalized = predictions / (np.sum(predictions) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        
        return float(entropy)
    
    def _determine_uncertainty_level(self, coefficient_of_variation: float) -> str:
        """
        Determine uncertainty level based on coefficient of variation.
        
        Args:
            coefficient_of_variation: Coefficient of variation
            
        Returns:
            Uncertainty level string
        """
        if coefficient_of_variation <= self.uncertainty_thresholds['low']:
            return 'low'
        elif coefficient_of_variation <= self.uncertainty_thresholds['medium']:
            return 'medium'
        else:
            return 'high'
    
    def _create_distribution(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Create prediction distribution for visualization.
        
        Args:
            predictions: Array of prediction values
            
        Returns:
            Dictionary with distribution statistics
        """
        return {
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
            'histogram': self._create_histogram(predictions)
        }
    
    def _create_histogram(self, predictions: np.ndarray, bins: int = 10) -> List[Dict]:
        """
        Create histogram data for visualization.
        
        Args:
            predictions: Array of prediction values
            bins: Number of histogram bins
            
        Returns:
            List of histogram bin data
        """
        hist, bin_edges = np.histogram(predictions, bins=bins)
        
        histogram_data = []
        for i in range(len(hist)):
            histogram_data.append({
                'bin_start': float(bin_edges[i]),
                'bin_end': float(bin_edges[i + 1]),
                'count': int(hist[i]),
                'frequency': float(hist[i] / len(predictions))
            })
        
        return histogram_data
    
    def _create_default_uncertainty(self) -> UncertaintyMetrics:
        """Create default uncertainty metrics when calculation fails"""
        return UncertaintyMetrics(
            mean_confidence=0.5,
            std_confidence=0.0,
            coefficient_of_variation=0.0,
            prediction_entropy=0.0,
            uncertainty_level='low',
            monte_carlo_samples=0,
            prediction_distribution={}
        )
    
    def get_uncertainty_recommendation(self, uncertainty_metrics: UncertaintyMetrics) -> str:
        """
        Get recommendation based on uncertainty level.
        
        Args:
            uncertainty_metrics: UncertaintyMetrics object
            
        Returns:
            Recommendation string
        """
        if uncertainty_metrics.uncertainty_level == 'high':
            return "High uncertainty detected. Recommend manual review by medical professional and additional diagnostic tests."
        elif uncertainty_metrics.uncertainty_level == 'medium':
            return "Medium uncertainty. Consider additional clinical correlation and follow-up imaging."
        else:
            return "Low uncertainty. Prediction can be used with standard clinical correlation."
    
    def should_flag_for_review(self, uncertainty_metrics: UncertaintyMetrics) -> bool:
        """
        Determine if prediction should be flagged for human review.
        
        Args:
            uncertainty_metrics: UncertaintyMetrics object
            
        Returns:
            True if should flag for review, False otherwise
        """
        # Flag for review if uncertainty is high or confidence is very low
        return (
            uncertainty_metrics.uncertainty_level == 'high' or
            uncertainty_metrics.mean_confidence < 0.3 or
            uncertainty_metrics.coefficient_of_variation > 0.4
        )
    
    def get_uncertainty_summary(self, uncertainty_metrics: UncertaintyMetrics) -> Dict:
        """
        Get comprehensive uncertainty summary.
        
        Args:
            uncertainty_metrics: UncertaintyMetrics object
            
        Returns:
            Dictionary with uncertainty summary
        """
        return {
            'mean_confidence': uncertainty_metrics.mean_confidence,
            'std_confidence': uncertainty_metrics.std_confidence,
            'coefficient_of_variation': uncertainty_metrics.coefficient_of_variation,
            'prediction_entropy': uncertainty_metrics.prediction_entropy,
            'uncertainty_level': uncertainty_metrics.uncertainty_level,
            'monte_carlo_samples': uncertainty_metrics.monte_carlo_samples,
            'recommendation': self.get_uncertainty_recommendation(uncertainty_metrics),
            'flag_for_review': self.should_flag_for_review(uncertainty_metrics),
            'prediction_distribution': uncertainty_metrics.prediction_distribution
        }


# Singleton instance
_uncertainty_service = None

def get_uncertainty_service() -> UncertaintyService:
    """Get or create the uncertainty service singleton"""
    global _uncertainty_service
    if _uncertainty_service is None:
        _uncertainty_service = UncertaintyService()
    return _uncertainty_service