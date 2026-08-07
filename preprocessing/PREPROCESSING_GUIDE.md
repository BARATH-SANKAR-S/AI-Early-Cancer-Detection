# Preprocessing Guide - Module 6

## Overview

This guide explains the complete preprocessing pipeline for both clinical data and medical images implemented in Module 6. Preprocessing is a critical step in machine learning that prepares raw data for model training by handling inconsistencies, noise, and formatting issues.

---

## Clinical Data Preprocessing

### Step 1: Missing Value Handling

**Purpose**: Handle missing or null values in the dataset to prevent errors during model training.

**Why It's Important**:
- Missing values can cause errors in mathematical operations
- Many machine learning algorithms cannot handle missing values
- Imputation preserves data that would otherwise be lost

**Methods Implemented**:

1. **Mean Imputation**: Fill numerical columns with the mean value
   - Best for normally distributed data
   - Preserves the mean of the distribution
   - Example: Age column with missing values filled with average age

2. **Median Imputation**: Fill numerical columns with the median value
   - Robust to outliers
   - Best for skewed distributions
   - Example: Income column with extreme values

3. **Mode Imputation**: Fill all columns with the most frequent value
   - Works for both numerical and categorical data
   - Preserves the most common category/value
   - Example: Gender column filled with most common gender

4. **Drop Rows**: Remove rows with any missing values
   - Simplest approach
   - Can lead to significant data loss
   - Use only when missing data is minimal

**Implementation**:
```python
df_cleaned = handle_missing_values(df, strategy='mean')
```

---

### Step 2: Duplicate Removal

**Purpose**: Remove duplicate rows to prevent data leakage and bias.

**Why It's Important**:
- Duplicate records can cause overfitting
- They artificially increase dataset size without adding information
- Can bias model training toward repeated patterns

**Methods Implemented**:
- Identify duplicates based on all columns or specified subset
- Remove all but the first occurrence of each duplicate
- Log number of duplicates removed for transparency

**Implementation**:
```python
df_cleaned = remove_duplicates(df, subset=['patient_id', 'date'])
```

---

### Step 3: Categorical Encoding

**Purpose**: Convert categorical variables to numerical values for machine learning algorithms.

**Why It's Important**:
- Most ML algorithms require numerical input
- Categorical data cannot be directly used in mathematical operations
- Encoding preserves categorical information in numerical form

**Methods Implemented**:

1. **Label Encoding**: Convert categories to integers (0, 1, 2, ...)
   - Efficient for ordinal data with natural ordering
   - Example: Low=0, Medium=1, High=2
   - Can introduce artificial ordering for nominal data

2. **One-Hot Encoding**: Create binary columns for each category
   - No artificial ordering introduced
   - Example: Gender becomes Gender_Male (0/1) and Gender_Female (0/1)
   - Increases dimensionality but preserves all information

**Implementation**:
```python
df_encoded = encode_categorical(df, target_column='diagnosis', encoding_method='label')
```

---

### Step 4: Feature Scaling

**Purpose**: Normalize numerical features to a consistent range.

**Why It's Important**:
- Features with different scales can bias distance-based algorithms
- Gradient descent converges faster with scaled features
- Prevents features with large ranges from dominating

**Methods Implemented**:

1. **Standard Scaling (Z-score)**: Mean=0, Std=1
   - Formula: (x - mean) / std
   - Best for normally distributed data
   - Preserves outliers but centers data

2. **MinMax Scaling**: Scale to [0, 1] range
   - Formula: (x - min) / (max - min)
   - Best for bounded data
   - Sensitive to outliers

**Implementation**:
```python
df_scaled = scale_features(df, target_column='diagnosis', scaling_method='standard')
```

---

### Step 5: Feature Selection

**Purpose**: Select the most relevant features to improve model performance and reduce overfitting.

**Why It's Important**:
- Reduces dimensionality and computational cost
- Removes irrelevant or redundant features
- Improves model interpretability
- Prevents overfitting by reducing noise

**Methods Implemented**:

1. **K-Best (ANOVA F-value)**: Select top K features using statistical test
   - Uses ANOVA F-value between feature and target
   - Fast and effective for classification
   - Example: Select top 10 most predictive features

2. **Mutual Information**: Select based on mutual information with target
   - Measures non-linear relationships
   - More robust than correlation
   - Captures complex dependencies

3. **Recursive Feature Elimination (RFE)**: Iteratively remove least important features
   - Uses Random Forest to rank features
   - Considers feature interactions
   - More computationally expensive but accurate

**Implementation**:
```python
X_selected, features = select_features(X, y, method='kbest', k=10)
```

---

### Step 6: Train-Test Split

**Purpose**: Split data into training and testing sets for model evaluation.

**Why It's Important**:
- Prevents data leakage (testing on training data)
- Provides unbiased evaluation of model performance
- Allows detection of overfitting

**Methods Implemented**:

1. **Stratified Split**: Maintains class distribution in both sets
   - Ensures balanced representation of classes
   - Critical for imbalanced datasets
   - Example: If 20% positive in full set, 20% positive in train/test

2. **Random Split**: Random division without stratification
   - Simpler but may create imbalanced splits
   - Suitable for balanced datasets

**Implementation**:
```python
X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, stratify=True)
```

---

## Medical Image Preprocessing

### Step 1: Image Loading

**Purpose**: Load medical images from file system into memory for processing.

**Why It's Important**:
- Converts image files to numerical arrays
- Standardizes input format for processing pipeline
- Handles various image formats (JPG, PNG, BMP, TIFF)

**Implementation**:
```python
image = Image.open('path/to/image.jpg')
image_array = np.array(image)
```

---

### Step 2: Resizing

**Purpose**: Resize images to a consistent size for model input.

**Why It's Important**:
- CNN models require fixed input dimensions
- Standardizes input across dataset
- Reduces computational cost with smaller sizes

**Methods Implemented**:

1. **Nearest Neighbor**: Fast but can cause aliasing
   - Uses nearest pixel value
   - Fastest method
   - Can produce blocky results

2. **Bilinear**: Good balance of speed and quality
   - Uses linear interpolation
   - Standard choice for most applications
   - Smooth results

3. **Bicubic**: Higher quality, slower
   - Uses cubic interpolation
   - Better for upscaling
   - More computationally expensive

4. **Lanczos**: Best quality, slowest
   - Uses Lanczos window
   - Highest quality results
   - Most computationally expensive

**Implementation**:
```python
resized = resize_image(image, target_size=(224, 224), method='bilinear')
```

---

### Step 3: Normalization

**Purpose**: Normalize pixel values to a consistent range.

**Why It's Important**:
- Standardizes pixel intensity across images
- Helps neural networks converge faster
- Reduces impact of varying lighting conditions

**Methods Implemented**:

1. **MinMax Normalization**: Scale to [0, 1]
   - Formula: (pixel - min) / (max - min)
   - Preserves relative intensity relationships
   - Standard for deep learning

2. **Z-score Normalization**: Mean=0, Std=1
   - Formula: (pixel - mean) / std
   - Centers data around zero
   - Useful for statistical methods

3. **Percentile Clipping**: Clip outliers then scale
   - Clips to 1st-99th percentile range
   - Robust to extreme values
   - Good for images with bright spots

**Implementation**:
```python
normalized = normalize_image(image, method='minmax')
```

---

### Step 4: Noise Removal

**Purpose**: Remove noise while preserving important diagnostic features.

**Why It's Important**:
- Medical images often contain acquisition noise
- Noise can interfere with feature extraction
- Denoising improves model performance

**Methods Implemented**:

1. **Gaussian Blur**: Smooths noise, fast
   - Applies Gaussian kernel
   - Good for general noise reduction
   - Can blur edges

2. **Median Filter**: Good for salt-and-pepper noise
   - Uses median of neighboring pixels
   - Preserves edges better than Gaussian
   - Effective for impulse noise

3. **Bilateral Filter**: Preserves edges while reducing noise
   - Considers both spatial and intensity differences
   - Best for preserving fine details
   - Slower than Gaussian

4. **Non-Local Means**: Advanced, preserves fine details
   - Uses similar patches across image
   - Best quality denoising
   - Most computationally expensive

**Implementation**:
```python
denoised = remove_noise(image, method='bilateral')
```

---

### Step 5: Contrast Enhancement

**Purpose**: Improve image contrast for better feature extraction.

**Why It's Important**:
- Medical images often have low contrast
- Enhancement makes features more visible
- Improves model's ability to detect patterns

**Methods Implemented**:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   - Adapts enhancement to local regions
   - Limits contrast to prevent noise amplification
   - Best for medical images
   - Standard choice for radiology

2. **Histogram Equalization**: Standard histogram equalization
   - Spreads intensity values evenly
   - Can over-enhance noise
   - Global enhancement

3. **Gamma Correction**: Non-linear brightness adjustment
   - Formula: output = input^gamma
   - Gamma > 1: Darkens image
   - Gamma < 1: Brightens image

**Implementation**:
```python
enhanced = enhance_contrast(image, method='clahe')
```

---

### Step 6: Data Augmentation

**Purpose**: Generate additional training samples to prevent overfitting.

**Why It's Important**:
- Increases effective dataset size
- Teaches model invariance to transformations
- Reduces overfitting on limited data
- Improves generalization

**Methods Implemented**:

1. **Horizontal Flip**: Mirror image horizontally
   - Doubles dataset size
   - Natural symmetry for many medical images
   - Fast operation

2. **Vertical Flip**: Mirror image vertically
   - Useful for certain imaging modalities
   - Less common than horizontal flip

3. **Rotation**: Random rotation (-15° to +15°)
   - Teaches rotational invariance
   - Small angles preserve anatomy
   - Random angle for variety

4. **Brightness**: Random brightness adjustment
   - Simulates different lighting conditions
   - Factor: 0.7 to 1.3
   - Improves robustness

5. **Contrast**: Random contrast adjustment
   - Simulates different contrast settings
   - Factor: 0.7 to 1.3
   - Improves robustness

6. **Zoom**: Random zoom (crop and resize)
   - Teaches scale invariance
   - Zoom factor: 0.8 to 1.0
   - Simulates different distances

7. **Noise**: Add Gaussian noise
   - Teaches noise robustness
   - Small noise level (std=0.02)
   - Prevents overfitting to clean images

**Implementation**:
```python
augmented = augment_image(image, augmentations=['horizontal_flip', 'rotation', 'brightness'])
```

---

## Usage Examples

### Clinical Data Preprocessing

```bash
python preprocessing/clinical/clinical_preprocessor.py \
    --input datasets/raw/clinical_data.csv \
    --target diagnosis \
    --output-dir datasets/processed \
    --missing-strategy mean \
    --encoding label \
    --scaling standard \
    --feature-selection kbest \
    --n-features 10 \
    --test-size 0.2
```

### Medical Image Preprocessing

```bash
python preprocessing/images/image_preprocessor.py \
    --input datasets/skin_cancer/isic_dicm_17k \
    --output datasets/processed/images \
    --target-size 224 224 \
    --normalize minmax \
    --denoise bilateral \
    --enhance clahe \
    --augment \
    --augmentations horizontal_flip rotation brightness
```

---

## Best Practices

1. **Always validate preprocessing steps** on a small sample before full pipeline
2. **Save preprocessing artifacts** (scalers, encoders) for consistent inference
3. **Use stratified splits** for imbalanced datasets
4. **Apply same preprocessing** to training and test data
5. **Document all parameters** used for reproducibility
6. **Visualize preprocessed data** to verify quality
7. **Handle class imbalance** during preprocessing if needed
8. **Use appropriate methods** for data type (ordinal vs nominal)

---

## Common Issues and Solutions

**Issue**: Too many missing values
- **Solution**: Consider dropping columns with >50% missing values

**Issue**: High cardinality categorical features
- **Solution**: Use target encoding or frequency encoding instead of one-hot

**Issue**: Outliers affecting scaling
- **Solution**: Use robust scaling or percentile clipping

**Issue**: Images too large for memory
- **Solution**: Resize to smaller dimensions or use batch processing

**Issue**: Augmentation changes diagnostic features
- **Solution**: Use conservative augmentation angles and factors

**Issue**: Different image formats in dataset
- **Solution**: Convert all to consistent format (PNG) during preprocessing
