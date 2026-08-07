"""
Clinical Data Preprocessor for Module 6
Handles missing values, duplicates, encoding, scaling, feature selection, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClinicalDataPreprocessor:
    """
    Complete preprocessing pipeline for clinical data
    
    Steps:
    1. Missing Value Handling
    2. Duplicate Removal
    3. Encoding (Categorical to Numerical)
    4. Scaling (Numerical Features)
    5. Feature Selection
    6. Train-Test Split
    """
    
    def __init__(self, output_dir: str = 'datasets/processed'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store preprocessing artifacts
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.selected_features = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        logger.info(f"Missing values before handling:\n{missing_counts[missing_counts > 0]}")
        
        df_cleaned = df.copy()
        
        if strategy == 'drop':
            # Drop rows with any missing values
            df_cleaned = df_cleaned.dropna()
            logger.info(f"Dropped {len(df) - len(df_cleaned)} rows with missing values")
            
        elif strategy == 'mean':
            # Fill numerical columns with mean
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    
            # Fill categorical columns with mode
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    
        elif strategy == 'median':
            # Fill numerical columns with median
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    
            # Fill categorical columns with mode
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    
        elif strategy == 'mode':
            # Fill all columns with mode
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        # Verify no missing values remain
        remaining_missing = df_cleaned.isnull().sum().sum()
        logger.info(f"Missing values after handling: {remaining_missing}")
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicate detection (None = all columns)
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=subset)
        duplicates_removed = initial_count - len(df_cleaned)
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        logger.info(f"Rows before: {initial_count}, Rows after: {len(df_cleaned)}")
        
        return df_cleaned
    
    def encode_categorical(self, df: pd.DataFrame, target_column: str, 
                          encoding_method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables to numerical values
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (not encoded)
            encoding_method: 'label' or 'onehot'
            
        Returns:
            DataFrame with categorical variables encoded
        """
        logger.info(f"Encoding categorical variables using {encoding_method} encoding")
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Exclude target column from encoding
        if target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        logger.info(f"Categorical columns to encode: {list(categorical_cols)}")
        
        if encoding_method == 'label':
            # Label encoding for each categorical column
            for col in categorical_cols:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
                logger.info(f"Label encoded {col}: {len(encoder.classes_)} classes")
                
        elif encoding_method == 'onehot':
            # One-hot encoding for categorical columns
            for col in categorical_cols:
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded = encoder.fit_transform(df_encoded[[col]])
                
                # Create column names for one-hot encoded features
                feature_names = [f"{col}_{i}" for i in range(encoded.shape[1])]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_encoded.index)
                
                # Drop original column and add encoded columns
                df_encoded = df_encoded.drop(col, axis=1)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
                self.encoders[col] = encoder
                logger.info(f"One-hot encoded {col}: {encoded.shape[1]} features created")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, target_column: str, 
                      scaling_method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (not scaled)
            scaling_method: 'standard' or 'minmax'
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {scaling_method} scaling")
        
        df_scaled = df.copy()
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        # Exclude target column from scaling
        if target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)
        
        logger.info(f"Numerical columns to scale: {list(numerical_cols)}")
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}. Using standard scaling.")
            scaler = StandardScaler()
        
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        self.scalers['feature_scaler'] = scaler
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return df_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'kbest', 
                       k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most important features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('kbest', 'mutual_info', 'rfe')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        logger.info(f"Selecting features using {method} method (k={k})")
        
        if method == 'kbest':
            # Select K Best features using ANOVA F-value
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'mutual_info':
            # Select K Best features using mutual information
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Recursive Feature Elimination with Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(rf, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using all features.")
            X_selected = X.values
            selected_features = X.columns.tolist()
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected_df, selected_features
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                            test_size: float = 0.2, random_state: int = 42,
                            stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify the split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data into train/test (test_size={test_size})")
        
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Testing set: {len(X_test)} samples")
        
        # Print class distribution
        logger.info(f"Training class distribution:\n{y_train.value_counts()}")
        logger.info(f"Testing class distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str,
                          missing_strategy: str = 'mean',
                          encoding_method: str = 'label',
                          scaling_method: str = 'standard',
                          feature_selection_method: str = 'kbest',
                          n_features: int = 10,
                          test_size: float = 0.2) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            missing_strategy: Strategy for missing values
            encoding_method: Method for encoding categorical variables
            scaling_method: Method for scaling features
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary containing preprocessed data and artifacts
        """
        logger.info("Starting complete preprocessing pipeline")
        logger.info(f"Initial data shape: {df.shape}")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Encode categorical variables
        df = self.encode_categorical(df, target_column, encoding_method=encoding_method)
        
        # Step 4: Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Step 5: Scale features
        X = self.scale_features(X, target_column, scaling_method=scaling_method)
        
        # Step 6: Feature selection
        X, selected_features = self.select_features(X, y, method=feature_selection_method, k=n_features)
        
        # Step 7: Train-test split
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y, test_size=test_size)
        
        # Store results
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'selected_features': selected_features,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_selector': self.feature_selector
        }
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return results
    
    def save_preprocessed_data(self, results: Dict, output_prefix: str = 'clinical'):
        """
        Save preprocessed data and artifacts
        
        Args:
            results: Dictionary containing preprocessed data
            output_prefix: Prefix for output files
        """
        logger.info(f"Saving preprocessed data with prefix: {output_prefix}")
        
        # Save train/test data
        results['X_train'].to_csv(self.output_dir / f'{output_prefix}_X_train.csv', index=False)
        results['X_test'].to_csv(self.output_dir / f'{output_prefix}_X_test.csv', index=False)
        results['y_train'].to_csv(self.output_dir / f'{output_prefix}_y_train.csv', index=False)
        results['y_test'].to_csv(self.output_dir / f'{output_prefix}_y_test.csv', index=False)
        
        # Save preprocessing artifacts
        with open(self.output_dir / f'{output_prefix}_scalers.pkl', 'wb') as f:
            pickle.dump(results['scalers'], f)
        
        with open(self.output_dir / f'{output_prefix}_encoders.pkl', 'wb') as f:
            pickle.dump(results['encoders'], f)
        
        with open(self.output_dir / f'{output_prefix}_feature_selector.pkl', 'wb') as f:
            pickle.dump(results['feature_selector'], f)
        
        # Save selected features
        with open(self.output_dir / f'{output_prefix}_selected_features.txt', 'w') as f:
            f.write('\n'.join(results['selected_features']))
        
        logger.info(f"Preprocessed data saved to {self.output_dir}")


def main():
    """Main function to test clinical data preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess clinical data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--output-dir', type=str, default='datasets/processed', help='Output directory')
    parser.add_argument('--missing-strategy', type=str, default='mean', 
                       choices=['mean', 'median', 'mode', 'drop'])
    parser.add_argument('--encoding', type=str, default='label', choices=['label', 'onehot'])
    parser.add_argument('--scaling', type=str, default='standard', choices=['standard', 'minmax'])
    parser.add_argument('--feature-selection', type=str, default='kbest', 
                       choices=['kbest', 'mutual_info', 'rfe'])
    parser.add_argument('--n-features', type=int, default=10, help='Number of features to select')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = ClinicalDataPreprocessor(output_dir=args.output_dir)
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(
        df=df,
        target_column=args.target,
        missing_strategy=args.missing_strategy,
        encoding_method=args.encoding,
        scaling_method=args.scaling,
        feature_selection_method=args.feature_selection,
        n_features=args.n_features,
        test_size=args.test_size
    )
    
    # Save results
    preprocessor.save_preprocessed_data(results, output_prefix='clinical')
    
    logger.info("Clinical data preprocessing completed successfully")


if __name__ == '__main__':
    main()
