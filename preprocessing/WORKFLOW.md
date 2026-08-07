# Preprocessing Workflow Diagram - Module 6

## Clinical Data Preprocessing Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLINICAL DATA PREPROCESSING                      │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Raw Data    │
│  (CSV File)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: MISSING VALUE HANDLING                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  • Identify missing values                                              │
│  • Strategy Options:                                                    │
│    - Mean: Fill numerical columns with mean                            │
│    - Median: Fill numerical columns with median                        │
│    - Mode: Fill all columns with mode                                  │
│    - Drop: Remove rows with missing values                             │
│  • Verify no missing values remain                                      │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DUPLICATE REMOVAL                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Identify duplicate rows                                              │
│  • Remove duplicates based on specified columns                        │
│  • Log number of duplicates removed                                    │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: CATEGORICAL ENCODING                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  • Identify categorical columns                                         │
│  • Encoding Options:                                                    │
│    - Label Encoding: Convert categories to integers (0,1,2...)         │
│    - One-Hot Encoding: Create binary columns for each category          │
│  • Exclude target column from encoding                                  │
│  • Store encoders for later use                                         │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: FEATURE SCALING                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  • Identify numerical columns                                           │
│  • Scaling Options:                                                     │
│    - Standard Scaling: Mean=0, Std=1 (Z-score normalization)           │
│    - MinMax Scaling: Scale to [0, 1] range                             │
│  • Exclude target column from scaling                                   │
│  • Store scaler for later use                                           │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: FEATURE SELECTION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Separate features (X) and target (y)                                 │
│  • Selection Methods:                                                   │
│    - K-Best: Select top K features using ANOVA F-value                 │
│    - Mutual Info: Select based on mutual information with target         │
│    - RFE: Recursive Feature Elimination with Random Forest               │
│  • Store selected feature names                                         │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: TRAIN-TEST SPLIT                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  • Split data into training and testing sets                            │
│  • Options:                                                             │
│    - Stratified: Maintain class distribution in both sets                │
│    - Random: Random split without stratification                        │
│  • Typical split: 80% train, 20% test                                   │
│  • Log class distribution in each set                                   │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  X_train     │
│  X_test      │
│  y_train     │
│  y_test      │
└──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SAVE PREPROCESSED DATA                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  • Save train/test splits as CSV files                                  │
│  • Save scalers, encoders, feature selectors as pickle files             │
│  • Save selected feature names as text file                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Medical Image Preprocessing Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEDICAL IMAGE PREPROCESSING                          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Raw Image    │
│  (File/Dir)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: IMAGE LOADING                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  • Load image from file path or directory                               │
│  • Support formats: JPG, JPEG, PNG, BMP, TIFF                           │
│  • Convert to numpy array for processing                                │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: RESIZING                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  • Resize to target size (e.g., 224x224 for CNN models)                 │
│  • Resampling Methods:                                                  │
│    - Nearest: Fast but can cause aliasing                                │
│    - Bilinear: Good balance of speed and quality                         │
│    - Bicubic: Higher quality, slower                                     │
│    - Lanczos: Best quality, slowest                                     │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: NORMALIZATION                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  • Normalize pixel values to consistent range                            │
│  • Normalization Methods:                                                │
│    - MinMax: Scale to [0, 1] range                                    │
│    - Z-score: Mean=0, Std=1 (standard normal distribution)              │
│    - Percentile: Clip outliers (1st-99th percentile) then scale          │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: NOISE REMOVAL                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  • Remove noise while preserving important features                     │
│  • Denoising Methods:                                                    │
│    - Gaussian Blur: Smooths noise, fast                                │
│    - Median Filter: Good for salt-and-pepper noise                       │
│    - Bilateral Filter: Preserves edges while reducing noise             │
│    - Non-Local Means: Advanced, preserves fine details                  │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: CONTRAST ENHANCEMENT                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  • Improve image contrast for better feature extraction                  │
│  • Enhancement Methods:                                                  │
│    - CLAHE: Contrast Limited Adaptive Histogram Equalization             │
│    - Histogram Eq: Standard histogram equalization                        │
│    - Gamma Correction: Non-linear brightness adjustment                  │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: DATA AUGMENTATION (Optional)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  • Generate additional training samples to prevent overfitting          │
│  • Augmentation Techniques:                                              │
│    - Horizontal Flip: Mirror image horizontally                         │
│    - Vertical Flip: Mirror image vertically                             │
│    - Rotation: Random rotation (-15° to +15°)                           │
│    - Brightness: Random brightness adjustment                           │
│    - Contrast: Random contrast adjustment                                │
│    - Zoom: Random zoom (crop and resize)                               │
│    - Noise: Add Gaussian noise                                         │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Preprocessed │
│  Images      │
│  (Numpy)     │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SAVE PREPROCESSED IMAGES                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Save as PNG files in output directory                                │
│  • Naming convention: originalname_processedN.png                      │
│  • Maintain original data type (uint8) for compatibility               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Combined Preprocessing Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMBINED PREPROCESSING PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐          ┌──────────────────┐
│  Clinical Data   │          │  Medical Images  │
│  (CSV/Database)  │          │  (Image Files)   │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Missing Value       │    │ Image Loading       │
│ Handling            │    │                     │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Duplicate Removal    │    │ Resizing            │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Categorical Encoding │    │ Normalization       │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Feature Scaling     │    │ Noise Removal       │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Feature Selection    │    │ Contrast Enhancement │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Train-Test Split     │    │ Data Augmentation    │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ X_train, X_test      │    │ Preprocessed Images │
│ y_train, y_test      │    │ (Numpy Arrays)      │
└────────┬─────────────┘    └────────┬─────────────┘
         │                             │
         └─────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  SAVE PREPROCESSED DATA     │
        │  • CSV files for clinical   │
        │  • PNG files for images     │
        │  • Pickle files for models  │
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  READY FOR MODEL TRAINING   │
        └──────────────────────────────┘
```

## Preprocessing Parameters Summary

### Clinical Data Parameters
- **Missing Value Strategy**: mean, median, mode, drop
- **Encoding Method**: label, onehot
- **Scaling Method**: standard, minmax
- **Feature Selection Method**: kbest, mutual_info, rfe
- **Number of Features**: K (user-defined)
- **Test Size**: 0.2 (20% for testing)

### Medical Image Parameters
- **Target Size**: (224, 224) for CNN models
- **Normalization Method**: minmax, zscore, percentile
- **Denoising Method**: gaussian, median, bilateral, non_local_means
- **Enhancement Method**: clahe, histogram_eq, gamma
- **Augmentation**: horizontal_flip, vertical_flip, rotation, brightness, contrast, zoom, noise

## File Organization

```
preprocessing/
├── clinical/
│   └── clinical_preprocessor.py    # Clinical data preprocessing
├── images/
│   └── image_preprocessor.py        # Medical image preprocessing
├── utils/
│   └── (utility functions)          # Helper functions
└── WORKFLOW.md                      # This workflow diagram

datasets/processed/
├── clinical_X_train.csv            # Preprocessed clinical features (train)
├── clinical_X_test.csv             # Preprocessed clinical features (test)
├── clinical_y_train.csv            # Clinical labels (train)
├── clinical_y_test.csv             # Clinical labels (test)
├── clinical_scalers.pkl            # Saved scalers
├── clinical_encoders.pkl           # Saved encoders
├── clinical_feature_selector.pkl   # Saved feature selector
├── clinical_selected_features.txt  # Selected feature names
└── images/                         # Preprocessed images directory
```
