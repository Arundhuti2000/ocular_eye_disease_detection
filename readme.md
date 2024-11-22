# Eye Disease Detection Using Deep Learning

## Project Overview

This project implements a deep learning solution for detecting various eye diseases from fundus images. The project includes comprehensive data analysis, preprocessing, and multi-label classification of eye conditions.

### Key Features

- Multi-label classification of 8 different eye conditions
- Comprehensive data preprocessing and analysis
- Custom data augmentation pipeline
- Implementation of various deep learning architectures
- Detailed visualization of results and model performance

## Dataset

The project uses the ODIR-5K dataset (Ocular Disease Intelligent Recognition) which contains:

- 6,392 fundus images from both left and right eyes
- 8 disease categories including:
  - Normal (N)
  - Diabetes (D)
  - Glaucoma (G)
  - Cataract (C)
  - Age-related Macular Degeneration (A)
  - Hypertension (H)
  - Myopia (M)
  - Other diseases/abnormalities (O)

## Analysis Pipeline

### 1. Data Preprocessing

- Removal of low-quality images
- Handling duplicates and inconsistencies
- Standardization of image sizes
- Data cleaning and validation

### 2. Exploratory Data Analysis

- Distribution analysis of eye conditions
- Age and gender analysis across different conditions
- Correlation analysis between different eye diseases
- Visualization of disease patterns

### 3. Data Classification

- Custom classification system for eye diseases
- Structured organization of images into disease categories
- Implementation of balanced sampling strategies

## Technical Implementation

### Dependencies

```python
- numpy
- pandas
- tensorflow
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- plotly
```

### Key Findings

1. Disease Distribution

   - Analysis of normal vs. abnormal fundus images
   - Distribution of diseases across age groups
   - Gender-based disease patterns

2. Data Patterns
   - Correlation between left and right eye conditions
   - Age-related disease patterns
   - Gender-specific disease prevalence

## Acknowledgments

- Initial analysis framework inspired by kaggel community work
- Custom implementation of multi-label classification
- Original dataset: ODIR-5K (Ocular Disease Intelligent Recognition)

## Model Performance

### Classification Results

The model achieved promising results across different eye conditions, with particularly strong performance in several categories.

#### Performance Metrics

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Normal       | 1.00      | 1.00   | 1.00     | 210     |
| Diabetes     | 0.67      | 0.88   | 0.76     | 205     |
| Glaucoma     | 0.83      | 0.85   | 0.84     | 34      |
| Cataract     | 0.88      | 1.00   | 0.93     | 35      |
| AMD          | 0.84      | 0.93   | 0.88     | 28      |
| Hypertension | 0.40      | 1.00   | 0.57     | 8       |
| Myopia       | 0.90      | 1.00   | 0.95     | 28      |
| Other        | 0.00      | 0.00   | 0.00     | 90      |

### Key Performance Highlights

- **Overall Accuracy**: 81% across all classes
- **Best Performing Categories**:
  - Normal (F1: 1.00)
  - Myopia (F1: 0.95)
  - Cataract (F1: 0.93)
  - AMD (F1: 0.88)
- **Challenges**:
  - 'Other' category (F1: 0.00)
  - Hypertension (Precision: 0.40)

### Performance Analysis

1. **Strong Performance**:

   - Perfect classification for Normal cases
   - Excellent detection of Myopia and Cataract
   - High recall across most categories

2. **Areas for Improvement**:

   - Poor performance in 'Other' category
   - Low precision in Hypertension detection
   - Moderate precision in Diabetes classification

3. **Class Imbalance**:
   - Large variation in support sizes (8 to 210 samples)
   - May affect model performance on minority classes

### Model Metrics Summary

- Macro Average F1-Score: 0.74
- Weighted Average F1-Score: 0.76
- Overall Accuracy: 0.81

This performance analysis suggests strong potential for clinical application while highlighting specific areas for future improvement, particularly in handling the 'Other' category and improving precision for Hypertension detection.

## Model Visualization and Analysis

### 1. Distribution of Fundus Conditions

![Fundus Distribution](./graphs/fundus_distribution.png)

Distribution of eye conditions across the dataset:

- Both Abnormal: 45.7% of cases
- Both Normal: 30.0% of cases
- Left Normal, Right Abnormal: 12.9%
- Right Normal, Left Abnormal: 11.4%

This indicates a significant presence of bilateral conditions in the dataset.

### 2. Disease Cluster Analysis

![Disease Clusters](./graphs/disease_cluster.png)

Principal Component Analysis (PCA) visualization shows the relationship between different eye conditions:

- Clear separation between Normal (N) and Diabetic (D) cases
- Glaucoma (G) shows distinct clustering
- Age-related conditions (A) and Normal cases show some overlap
- Patient Age is centrally positioned, indicating its relevance across conditions

### 3. Confusion Matrix

![Confusion Matrix](./graphs/confusion_matrix.png)

The confusion matrix demonstrates the model's classification performance across different eye conditions:

- Perfect prediction (210/210) for Normal class
- Strong performance for Diabetes (181 correct predictions)
- Excellent accuracy for specialized conditions (Cataract: 35/35, Myopia: 28/28)
- Some misclassifications between Diabetes and Other categories

### 4. Model Training Performance

![Model Performance](./graphs//model_accuracy_loss.png)

Training metrics over epochs show:

- **Accuracy**:

  - Validation accuracy reaches ~80%
  - Training accuracy stabilizes around 65%
  - Good convergence without significant overfitting

- **Loss**:
  - Both training and validation loss decrease steadily
  - Model shows stable learning with minor fluctuations
  - Optimal convergence achieved around epoch 15

These visualizations demonstrate:

1. Strong classification performance across major disease categories
2. Clear disease clustering patterns
3. Balanced distribution of normal and abnormal cases
4. Stable and effective model training process

## Contributors

- Arundhati Das & Pranay Ghuge

---

**Note**: This project combines both established analysis methods and custom implementations for multi-label classification of eye diseases.
