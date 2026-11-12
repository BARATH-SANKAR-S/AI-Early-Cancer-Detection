"""
Federated Learning Simulation Service
Simulates federated learning across multiple hospitals/institutions.
Enables privacy-preserving model training through distributed learning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HospitalRole(Enum):
    """Types of hospital roles in federated learning"""
    CENTRAL_SERVER = "central_server"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


@dataclass
class HospitalClient:
    """Represents a hospital participating in federated learning"""
    hospital_id: str
    hospital_name: str
    role: HospitalRole
    data_size: int
    local_model_accuracy: float
    contribution_weight: float
    last_sync: datetime
    privacy_level: str  # low, medium, high


@dataclass
class FederatedRound:
    """Data for a single federated learning round"""
    round_number: int
    participating_hospitals: List[str]
    global_model_accuracy: float
    model_delta: Dict
    convergence_metrics: Dict
    privacy_loss: float
    timestamp: datetime


@dataclass
class TrainingProgress:
    """Overall federated learning progress"""
    total_rounds: int
    current_round: int
    best_accuracy: float
    convergence_rate: float
    privacy_budget_remaining: float
    total_data_samples: int


class FederatedLearningService:
    """
    Service for simulating federated learning across multiple hospitals.
    Enables privacy-preserving model training through distributed learning.
    """
    
    def __init__(self):
        """Initialize the federated learning service"""
        self.hospital_clients = self._initialize_hospital_clients()
        self.federated_rounds = []
        self.current_round = 0
        self.global_model_params = self._initialize_global_model()
        self.privacy_budget = 1.0  # Privacy budget (epsilon-differential privacy)
        self.convergence_threshold = 0.95  # Target accuracy for convergence
        self.max_rounds = 50  # Maximum federated learning rounds
    
    def _initialize_hospital_clients(self) -> List[HospitalClient]:
        """Initialize hospital clients for federated learning"""
        hospitals = [
            HospitalClient(
                hospital_id="H001",
                hospital_name="Erode Government Hospital",
                role=HospitalRole.PARTICIPANT,
                data_size=1500,
                local_model_accuracy=0.72,
                contribution_weight=0.25,
                last_sync=datetime.utcnow(),
                privacy_level="high"
            ),
            HospitalClient(
                hospital_id="H002",
                hospital_name="Bhavani Medical Center",
                role=HospitalRole.PARTICIPANT,
                data_size=1200,
                local_model_accuracy=0.68,
                contribution_weight=0.20,
                last_sync=datetime.utcnow(),
                privacy_level="medium"
            ),
            HospitalClient(
                hospital_id="H003",
                hospital_name="Gobichettipalayam Cancer Institute",
                role=HospitalRole.PARTICIPANT,
                data_size=1800,
                local_model_accuracy=0.75,
                contribution_weight=0.30,
                last_sync=datetime.utcnow(),
                privacy_level="high"
            ),
            HospitalClient(
                hospital_id="H004",
                hospital_name="Sathyamangalam General Hospital",
                role=HospitalRole.PARTICIPANT,
                data_size=900,
                local_model_accuracy=0.65,
                contribution_weight=0.15,
                last_sync=datetime.utcnow(),
                privacy_level="low"
            ),
            HospitalClient(
                hospital_id="H005",
                hospital_name="Dharapuram Specialty Clinic",
                role=HospitalRole.OBSERVER,
                data_size=600,
                local_model_accuracy=0.70,
                contribution_weight=0.10,
                last_sync=datetime.utcnow(),
                privacy_level="medium"
            )
        ]
        return hospitals
    
    def _initialize_global_model(self) -> Dict:
        """Initialize global model parameters"""
        return {
            'weights': np.random.randn(1000).tolist(),  # Simulated model weights
            'accuracy': 0.70,  # Initial global accuracy
            'version': 'v1.0',
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def run_federated_round(self) -> FederatedRound:
        """
        Run a single round of federated learning.
        
        Returns:
            FederatedRound object with round results
        """
        if self.current_round >= self.max_rounds:
            logger.warning("Maximum federated learning rounds reached")
            return self._create_failed_round()
        
        # Select participating hospitals
        participating_hospitals = self._select_participants()
        
        # Simulate local training at each hospital
        local_updates = []
        for hospital in participating_hospitals:
            local_update = self._simulate_local_training(hospital)
            local_updates.append(local_update)
        
        # Aggregate local updates (Federated Averaging)
        aggregated_update = self._federated_averaging(local_updates)
        
        # Update global model
        self._update_global_model(aggregated_update)
        
        # Calculate privacy loss
        privacy_loss = self._calculate_privacy_loss(participating_hospitals)
        self.privacy_budget -= privacy_loss
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics()
        
        # Create federated round record
        federated_round = FederatedRound(
            round_number=self.current_round + 1,
            participating_hospitals=[h.hospital_id for h in participating_hospitals],
            global_model_accuracy=self.global_model_params['accuracy'],
            model_delta=aggregated_update,
            convergence_metrics=convergence_metrics,
            privacy_loss=privacy_loss,
            timestamp=datetime.utcnow()
        )
        
        self.federated_rounds.append(federated_round)
        self.current_round += 1
        
        logger.info(f"Federated round {self.current_round} completed with {len(participating_hospitals)} participants")
        
        return federated_round
    
    def _select_participants(self) -> List[HospitalClient]:
        """Select hospitals to participate in current round"""
        # Select hospitals with PARTICIPANT role
        participants = [h for h in self.hospital_clients if h.role == HospitalRole.PARTICIPANT]
        
        # Randomly select subset (simulation of client selection)
        num_participants = max(2, len(participants) // 2)
        selected_indices = np.random.choice(len(participants), num_participants, replace=False)
        
        return [participants[i] for i in selected_indices]
    
    def _simulate_local_training(self, hospital: HospitalClient) -> Dict:
        """
        Simulate local training at a hospital.
        
        Args:
            hospital: HospitalClient object
            
        Returns:
            Dictionary with local model update
        """
        # Simulate training improvement
        improvement = np.random.uniform(0.01, 0.05)
        new_accuracy = min(hospital.local_model_accuracy + improvement, 0.95)
        
        # Generate model delta (simulated weight updates)
        model_delta = {
            'hospital_id': hospital.hospital_id,
            'weight_updates': np.random.randn(1000).tolist(),
            'accuracy_improvement': improvement,
            'new_accuracy': new_accuracy,
            'data_size': hospital.data_size,
            'privacy_noise': self._add_privacy_noise(hospital.privacy_level)
        }
        
        # Update hospital's local accuracy
        hospital.local_model_accuracy = new_accuracy
        hospital.last_sync = datetime.utcnow()
        
        return model_delta
    
    def _add_privacy_noise(self, privacy_level: str) -> float:
        """Add privacy noise based on hospital's privacy level"""
        noise_levels = {
            'low': 0.01,
            'medium': 0.05,
            'high': 0.10
        }
        return np.random.normal(0, noise_levels.get(privacy_level, 0.05))
    
    def _federated_averaging(self, local_updates: List[Dict]) -> Dict:
        """
        Aggregate local updates using Federated Averaging.
        
        Args:
            local_updates: List of local model updates
            
        Returns:
            Aggregated model update
        """
        # Calculate weighted average based on data size
        total_data_size = sum(update['data_size'] for update in local_updates)
        
        # Weighted average of weight updates
        weighted_updates = []
        for update in local_updates:
            weight = update['data_size'] / total_data_size
            weighted_update = [w * weight for w in update['weight_updates']]
            weighted_updates.append(weighted_update)
        
        # Sum weighted updates
        aggregated_weights = [sum(weights) for weights in zip(*weighted_updates)]
        
        # Calculate average accuracy improvement
        avg_improvement = sum(update['accuracy_improvement'] for update in local_updates) / len(local_updates)
        
        return {
            'aggregated_weights': aggregated_weights,
            'average_improvement': avg_improvement,
            'participating_hospitals': len(local_updates),
            'total_data_size': total_data_size
        }
    
    def _update_global_model(self, aggregated_update: Dict):
        """Update global model with aggregated update"""
        # Update global weights
        self.global_model_params['weights'] = aggregated_update['aggregated_weights']
        
        # Update global accuracy
        current_accuracy = self.global_model_params['accuracy']
        improvement = aggregated_update['average_improvement']
        self.global_model_params['accuracy'] = min(current_accuracy + improvement, 0.98)
        
        # Update version and timestamp
        self.global_model_params['version'] = f"v{self.current_round + 1}.0"
        self.global_model_params['last_updated'] = datetime.utcnow().isoformat()
    
    def _calculate_privacy_loss(self, participants: List[HospitalClient]) -> float:
        """Calculate privacy loss for the round"""
        # Simple epsilon-differential privacy calculation
        num_participants = len(participants)
        base_loss = 0.01  # Base privacy loss per round
        participant_factor = 0.005 * num_participants
        
        return min(base_loss + participant_factor, 0.1)
    
    def _calculate_convergence_metrics(self) -> Dict:
        """Calculate convergence metrics"""
        if len(self.federated_rounds) < 2:
            return {
                'convergence_rate': 0.0,
                'accuracy_trend': 'stable',
                'is_converged': False
            }
        
        # Calculate accuracy improvement rate
        recent_rounds = self.federated_rounds[-5:] if len(self.federated_rounds) >= 5 else self.federated_rounds
        accuracies = [round.global_model_accuracy for round in recent_rounds]
        
        if len(accuracies) >= 2:
            improvement_rate = (accuracies[-1] - accuracies[0]) / len(accuracies)
        else:
            improvement_rate = 0.0
        
        # Determine trend
        if improvement_rate > 0.01:
            trend = 'improving'
        elif improvement_rate < -0.01:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        # Check convergence
        is_converged = self.global_model_params['accuracy'] >= self.convergence_threshold
        
        return {
            'convergence_rate': improvement_rate,
            'accuracy_trend': trend,
            'is_converged': is_converged,
            'target_accuracy': self.convergence_threshold
        }
    
    def _create_failed_round(self) -> FederatedRound:
        """Create a failed round object"""
        return FederatedRound(
            round_number=self.current_round,
            participating_hospitals=[],
            global_model_accuracy=self.global_model_params['accuracy'],
            model_delta={},
            convergence_metrics={'error': 'Maximum rounds reached'},
            privacy_loss=0.0,
            timestamp=datetime.utcnow()
        )
    
    def get_training_progress(self) -> TrainingProgress:
        """Get overall federated learning progress"""
        best_accuracy = max(
            [round.global_model_accuracy for round in self.federated_rounds],
            default=self.global_model_params['accuracy']
        )
        
        # Calculate convergence rate
        if len(self.federated_rounds) >= 2:
            first_accuracy = self.federated_rounds[0].global_model_accuracy
            last_accuracy = self.federated_rounds[-1].global_model_accuracy
            convergence_rate = (last_accuracy - first_accuracy) / len(self.federated_rounds)
        else:
            convergence_rate = 0.0
        
        total_data_samples = sum(hospital.data_size for hospital in self.hospital_clients)
        
        return TrainingProgress(
            total_rounds=self.max_rounds,
            current_round=self.current_round,
            best_accuracy=best_accuracy,
            convergence_rate=convergence_rate,
            privacy_budget_remaining=self.privacy_budget,
            total_data_samples=total_data_samples
        )
    
    def get_hospital_statistics(self) -> List[Dict]:
        """Get statistics for all participating hospitals"""
        return [
            {
                'hospital_id': hospital.hospital_id,
                'hospital_name': hospital.hospital_name,
                'role': hospital.role.value,
                'data_size': hospital.data_size,
                'local_model_accuracy': hospital.local_model_accuracy,
                'contribution_weight': hospital.contribution_weight,
                'privacy_level': hospital.privacy_level,
                'last_sync': hospital.last_sync.isoformat()
            }
            for hospital in self.hospital_clients
        ]
    
    def get_federated_learning_summary(self) -> Dict:
        """Get comprehensive federated learning summary"""
        progress = self.get_training_progress()
        hospital_stats = self.get_hospital_statistics()
        
        # Calculate round statistics
        if self.federated_rounds:
            avg_participants = np.mean([len(round.participating_hospitals) for round in self.federated_rounds])
            avg_privacy_loss = np.mean([round.privacy_loss for round in self.federated_rounds])
        else:
            avg_participants = 0
            avg_privacy_loss = 0.0
        
        return {
            'progress': {
                'total_rounds': progress.total_rounds,
                'current_round': progress.current_round,
                'completion_percentage': (progress.current_round / progress.total_rounds) * 100,
                'best_accuracy': progress.best_accuracy,
                'convergence_rate': progress.convergence_rate,
                'privacy_budget_remaining': progress.privacy_budget_remaining,
                'total_data_samples': progress.total_data_samples
            },
            'hospital_statistics': hospital_stats,
            'round_statistics': {
                'total_rounds_completed': len(self.federated_rounds),
                'average_participants_per_round': avg_participants,
                'average_privacy_loss_per_round': avg_privacy_loss,
                'current_global_accuracy': self.global_model_params['accuracy'],
                'current_model_version': self.global_model_params['version']
            },
            'convergence_status': self._calculate_convergence_metrics(),
            'privacy_status': {
                'privacy_budget_remaining': self.privacy_budget,
                'privacy_budget_used': 1.0 - self.privacy_budget,
                'privacy_status': 'adequate' if self.privacy_budget > 0.3 else 'low'
            }
        }
    
    def export_federated_data(self) -> Dict:
        """Export federated learning data for analysis"""
        return {
            'hospital_clients': self.get_hospital_statistics(),
            'federated_rounds': [
                {
                    'round_number': round.round_number,
                    'participating_hospitals': round.participating_hospitals,
                    'global_model_accuracy': round.global_model_accuracy,
                    'privacy_loss': round.privacy_loss,
                    'timestamp': round.timestamp.isoformat(),
                    'convergence_metrics': round.convergence_metrics
                }
                for round in self.federated_rounds
            ],
            'global_model': self.global_model_params,
            'training_progress': self.get_training_progress().__dict__
        }


# Singleton instance
_federated_learning_service = None

def get_federated_learning_service() -> FederatedLearningService:
    """Get or create the federated learning service singleton"""
    global _federated_learning_service
    if _federated_learning_service is None:
        _federated_learning_service = FederatedLearningService()
    return _federated_learning_service