"""
Geographic Cancer Risk Service
Provides cancer risk analysis and visualization for Erode district.
Generates geographic heatmaps and regional statistics.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RegionalStats:
    """Statistics for a specific region"""
    region_name: str
    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_confidence: float
    most_common_cancer_type: str
    trend_direction: str  # increasing, decreasing, stable


@dataclass
class GeographicHeatmapData:
    """Data for geographic heatmap visualization"""
    regions: List[Dict]
    overall_stats: Dict
    temporal_data: List[Dict]
    risk_factors: Dict


class GeographicService:
    """
    Service for geographic cancer risk analysis in Erode district.
    Provides regional statistics, heatmaps, and trend analysis.
    """
    
    # Erode district taluks (administrative divisions)
    ERODE_TALUKS = [
        "Erode", "Bhavani", "Gobichettipalayam", "Sathyamangalam",
        "Perundurai", "Kangeyam", "Dharapuram", "Kangayam",
        "Uttukuli", "Kodumudi", "Modakurichi", "Nambiyur"
    ]
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        RiskLevel.LOW: 0.0,
        RiskLevel.MEDIUM: 0.35,
        RiskLevel.HIGH: 0.70,
        RiskLevel.CRITICAL: 0.85
    }
    
    # Cancer type mappings for regional analysis
    CANCER_TYPE_REGIONS = {
        'lung': ['Erode', 'Bhavani', 'Perundurai'],  # Industrial areas
        'skin': ['Erode', 'Gobichettipalayam'],  # Urban areas
        'brain': ['Sathyamangalam', 'Dharapuram'],  # Rural areas
        'breast': ['Kangeyam', 'Kangayam', 'Uttukuli']  # Mixed areas
    }
    
    def __init__(self):
        """Initialize the geographic service"""
        self.mock_data_enabled = True  # Use mock data for demonstration
    
    def get_regional_statistics(self, region_name: str) -> Optional[RegionalStats]:
        """
        Get cancer statistics for a specific region.
        
        Args:
            region_name: Name of the region/taluk
            
        Returns:
            RegionalStats object or None if region not found
        """
        if region_name not in self.ERODE_TALUKS:
            logger.warning(f"Region {region_name} not found in Erode district")
            return None
        
        if self.mock_data_enabled:
            return self._generate_mock_regional_stats(region_name)
        
        # In production, this would query the database
        return self._get_regional_stats_from_db(region_name)
    
    def _generate_mock_regional_stats(self, region_name: str) -> RegionalStats:
        """Generate mock regional statistics for demonstration"""
        import random
        
        # Generate realistic mock data
        total_predictions = random.randint(50, 500)
        high_risk_count = int(total_predictions * random.uniform(0.1, 0.3))
        medium_risk_count = int(total_predictions * random.uniform(0.2, 0.4))
        low_risk_count = total_predictions - high_risk_count - medium_risk_count
        
        avg_confidence = random.uniform(0.6, 0.9)
        
        cancer_types = ['lung', 'skin', 'brain', 'breast']
        most_common_cancer_type = random.choice(cancer_types)
        
        trend_direction = random.choice(['increasing', 'decreasing', 'stable'])
        
        return RegionalStats(
            region_name=region_name,
            total_predictions=total_predictions,
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            avg_confidence=avg_confidence,
            most_common_cancer_type=most_common_cancer_type,
            trend_direction=trend_direction
        )
    
    def _get_regional_stats_from_db(self, region_name: str) -> Optional[RegionalStats]:
        """Get regional statistics from database (placeholder for production)"""
        # This would query PredictionHistory joined with Patient data
        # filtered by patient.district == region_name
        logger.info(f"Would query database for region: {region_name}")
        return None
    
    def generate_heatmap_data(self) -> GeographicHeatmapData:
        """
        Generate complete heatmap data for Erode district.
        
        Returns:
            GeographicHeatmapData with all visualization data
        """
        regions_data = []
        
        for taluk in self.ERODE_TALUKS:
            stats = self.get_regional_statistics(taluk)
            if stats:
                regions_data.append({
                    'name': taluk,
                    'total_predictions': stats.total_predictions,
                    'high_risk_count': stats.high_risk_count,
                    'medium_risk_count': stats.medium_risk_count,
                    'low_risk_count': stats.low_risk_count,
                    'avg_confidence': stats.avg_confidence,
                    'most_common_cancer_type': stats.most_common_cancer_type,
                    'trend_direction': stats.trend_direction,
                    'risk_score': self._calculate_regional_risk_score(stats)
                })
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(regions_data)
        
        # Generate temporal data (monthly trends)
        temporal_data = self._generate_temporal_data()
        
        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(regions_data)
        
        return GeographicHeatmapData(
            regions=regions_data,
            overall_stats=overall_stats,
            temporal_data=temporal_data,
            risk_factors=risk_factors
        )
    
    def _calculate_regional_risk_score(self, stats: RegionalStats) -> float:
        """
        Calculate overall risk score for a region.
        
        Args:
            stats: RegionalStats object
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        if stats.total_predictions == 0:
            return 0.0
        
        # Weight high-risk cases more heavily
        weighted_risk = (
            (stats.high_risk_count * 1.0) +
            (stats.medium_risk_count * 0.5) +
            (stats.low_risk_count * 0.1)
        ) / stats.total_predictions
        
        # Adjust by average confidence
        weighted_risk = weighted_risk * stats.avg_confidence
        
        # Apply trend multiplier
        trend_multiplier = {
            'increasing': 1.2,
            'decreasing': 0.8,
            'stable': 1.0
        }
        weighted_risk *= trend_multiplier.get(stats.trend_direction, 1.0)
        
        return min(weighted_risk, 1.0)
    
    def _calculate_overall_stats(self, regions_data: List[Dict]) -> Dict:
        """Calculate overall statistics for the entire district"""
        if not regions_data:
            return {}
        
        total_predictions = sum(r['total_predictions'] for r in regions_data)
        total_high_risk = sum(r['high_risk_count'] for r in regions_data)
        total_medium_risk = sum(r['medium_risk_count'] for r in regions_data)
        total_low_risk = sum(r['low_risk_count'] for r in regions_data)
        
        avg_confidence = sum(r['avg_confidence'] for r in regions_data) / len(regions_data)
        
        # Find most common cancer type across all regions
        cancer_type_counts = {}
        for region in regions_data:
            cancer_type = region['most_common_cancer_type']
            cancer_type_counts[cancer_type] = cancer_type_counts.get(cancer_type, 0) + 1
        
        most_common_cancer_type = max(cancer_type_counts, key=cancer_type_counts.get) if cancer_type_counts else 'unknown'
        
        return {
            'total_predictions': total_predictions,
            'total_high_risk': total_high_risk,
            'total_medium_risk': total_medium_risk,
            'total_low_risk': total_low_risk,
            'avg_confidence': avg_confidence,
            'most_common_cancer_type': most_common_cancer_type,
            'high_risk_percentage': (total_high_risk / total_predictions * 100) if total_predictions > 0 else 0,
            'medium_risk_percentage': (total_medium_risk / total_predictions * 100) if total_predictions > 0 else 0,
            'low_risk_percentage': (total_low_risk / total_predictions * 100) if total_predictions > 0 else 0
        }
    
    def _generate_temporal_data(self) -> List[Dict]:
        """Generate temporal data for trend analysis"""
        temporal_data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for month in range(12):
            current_date = base_date + timedelta(days=30 * month)
            import random
            
            # Generate mock monthly data
            total_predictions = random.randint(20, 100)
            high_risk_count = int(total_predictions * random.uniform(0.1, 0.3))
            
            temporal_data.append({
                'month': current_date.strftime('%Y-%m'),
                'total_predictions': total_predictions,
                'high_risk_count': high_risk_count,
                'avg_confidence': random.uniform(0.6, 0.9)
            })
        
        return temporal_data
    
    def _analyze_risk_factors(self, regions_data: List[Dict]) -> Dict:
        """Analyze risk factors across regions"""
        if not regions_data:
            return {}
        
        # Calculate risk factor correlations
        risk_factors = {
            'urban_areas': {
                'avg_risk_score': 0.0,
                'total_regions': 0
            },
            'rural_areas': {
                'avg_risk_score': 0.0,
                'total_regions': 0
            },
            'industrial_areas': {
                'avg_risk_score': 0.0,
                'total_regions': 0
            }
        }
        
        # Categorize regions and calculate averages
        urban_regions = ['Erode', 'Bhavani', 'Gobichettipalayam']
        industrial_regions = ['Erode', 'Perundurai', 'Bhavani']
        
        for region in regions_data:
            region_name = region['name']
            risk_score = region['risk_score']
            
            if region_name in urban_regions:
                risk_factors['urban_areas']['avg_risk_score'] += risk_score
                risk_factors['urban_areas']['total_regions'] += 1
            
            if region_name in industrial_regions:
                risk_factors['industrial_areas']['avg_risk_score'] += risk_score
                risk_factors['industrial_areas']['total_regions'] += 1
            else:
                risk_factors['rural_areas']['avg_risk_score'] += risk_score
                risk_factors['rural_areas']['total_regions'] += 1
        
        # Calculate averages
        for category in risk_factors:
            if risk_factors[category]['total_regions'] > 0:
                risk_factors[category]['avg_risk_score'] /= risk_factors[category]['total_regions']
        
        return risk_factors
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Determine risk level from risk score.
        
        Args:
            risk_score: Risk score between 0.0 and 1.0
            
        Returns:
            RiskLevel enum value
        """
        if risk_score >= self.RISK_THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_regional_recommendations(self, region_name: str) -> List[str]:
        """
        Get specific recommendations for a region based on risk analysis.
        
        Args:
            region_name: Name of the region
            
        Returns:
            List of recommendation strings
        """
        stats = self.get_regional_statistics(region_name)
        if not stats:
            return []
        
        risk_score = self._calculate_regional_risk_score(stats)
        risk_level = self.get_risk_level(risk_score)
        
        recommendations = []
        
        # Base recommendations
        recommendations.append("Regular cancer screening programs")
        recommendations.append("Public awareness campaigns about early detection")
        
        # Risk-specific recommendations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Increase mobile screening units in the area")
            recommendations.append("Prioritize specialist clinics and diagnostic centers")
            recommendations.append("Enhance emergency medical services")
            
            if stats.most_common_cancer_type == 'lung':
                recommendations.append("Anti-smoking campaigns and respiratory health programs")
            elif stats.most_common_cancer_type == 'skin':
                recommendations.append("Skin cancer awareness and UV protection education")
            elif stats.most_common_cancer_type == 'breast':
                recommendations.append("Breast cancer screening and mammography programs")
        
        # Trend-specific recommendations
        if stats.trend_direction == 'increasing':
            recommendations.append("Immediate intervention required due to rising trend")
            recommendations.append("Increase frequency of screening programs")
        elif stats.trend_direction == 'decreasing':
            recommendations.append("Continue current prevention strategies")
            recommendations.append("Monitor trend to ensure sustained improvement")
        
        return recommendations
    
    def export_heatmap_data(self, format: str = 'json') -> str:
        """
        Export heatmap data in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            String containing exported data
        """
        heatmap_data = self.generate_heatmap_data()
        
        if format == 'json':
            return json.dumps({
                'regions': heatmap_data.regions,
                'overall_stats': heatmap_data.overall_stats,
                'temporal_data': heatmap_data.temporal_data,
                'risk_factors': heatmap_data.risk_factors
            }, indent=2)
        elif format == 'csv':
            # Simple CSV export for regions data
            csv_lines = ['Region,Total Predictions,High Risk,Medium Risk,Low Risk,Avg Confidence,Risk Score']
            for region in heatmap_data.regions:
                csv_lines.append(
                    f"{region['name']},{region['total_predictions']},"
                    f"{region['high_risk_count']},{region['medium_risk_count']},"
                    f"{region['low_risk_count']},{region['avg_confidence']:.2f},"
                    f"{region['risk_score']:.2f}"
                )
            return '\n'.join(csv_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Singleton instance
_geographic_service = None

def get_geographic_service() -> GeographicService:
    """Get or create the geographic service singleton"""
    global _geographic_service
    if _geographic_service is None:
        _geographic_service = GeographicService()
    return _geographic_service