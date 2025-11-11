# Machine Learning Introduction - freeCodeCamp# FCC Intro to AI/ML Models and Probabilities

An introductory ipynb to basic models used in machine learning written for hands-on practice while learning through this [lecture video](https://youtu.be/i_LwzRVP7bg?list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s) by [Kylie Ying](https://www.youtube.com/c/YCubed)

This notebook provides a comprehensive introduction to various machine learning algorithms and techniques using Python. It covers both classification and regression problems with different approaches.



An introductory Jupyter notebook for hands-on practice while learning through this [lecture video](https://youtu.be/i_LwzRVP7bg?list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s) by [Kylie Ying](https://www.youtube.com/c/YCubed)Check out amazing lecture tutorials from [FreeCodeCamp](https://www.youtube.com/@freecodecamp)!

## Overview

The notebook is organized into two main sections:
1. **Classification Models** - Predicting gamma vs hadron particles using the Magic Telescope Dataset
2. **Regression Models** - Predicting bike rental counts using Seoul Bike Data

---

## Classification Models

### Dataset: Magic Gamma Telescope
Binary classification of particles from a gamma-ray telescope.

### 1. K-Nearest Neighbors (KNN)
- **Algorithm**: Instance-based learning that classifies data points based on the K nearest training examples
- **Use Case**: Non-parametric method suitable for multi-class classification
- **Hyperparameter**: `n_neighbors=5` (number of neighbors to consider)
- **Pros**: Simple, no training phase, works well with small datasets
- **Cons**: Computationally expensive during prediction, sensitive to irrelevant features

### 2. Naive Bayes
- **Algorithm**: Probabilistic classifier based on Bayes' theorem with conditional independence assumption
- **Use Case**: Fast classification, works well with high-dimensional data
- **Implementation**: Gaussian Naive Bayes for continuous features
- **Pros**: Fast, works well with text and categorical data, good baseline
- **Cons**: Assumes feature independence which is rarely true

### 3. Logistic Regression
- **Algorithm**: Linear model for binary classification using sigmoid activation function
- **Use Case**: Interpretable classification, probability predictions
- **Output**: Probability scores between 0 and 1
- **Pros**: Interpretable, efficient, works well with linear relationships
- **Cons**: Assumes linearity, may underfit complex patterns

### 4. Support Vector Machine (SVM)
- **Algorithm**: Finds optimal hyperplane that maximizes margin between classes
- **Use Case**: High-dimensional data, binary and multi-class classification
- **Kernel**: RBF (Radial Basis Function) by default
- **Pros**: Effective in high dimensions, memory efficient, versatile
- **Cons**: Slow training on large datasets, difficult to interpret

### 5. Neural Network (Classification)
- **Architecture**: Multi-layer perceptron with:
  - Input layer: 10 features
  - Hidden layer 1: Variable nodes (16/32/64) + ReLU + Dropout
  - Hidden layer 2: Variable nodes (16/32/64) + ReLU + Dropout
  - Output layer: 1 node with sigmoid activation (binary classification)
  
- **Hyperparameters Tuned via Grid Search**:
  - Number of nodes: [16, 32, 64]
  - Dropout probability: [0, 0.2]
  - Learning rate: [0.1, 0.005, 0.001]
  - Batch size: [32, 64, 128]
  
- **Pros**: Highly flexible, captures non-linear patterns, best for large datasets
- **Cons**: Requires more data, longer training time, harder to interpret

---

## Regression Models

### Dataset: Seoul Bike Rental Data
Predicts bike rental counts at noon based on environmental factors.

### 6. Simple Linear Regression
- **Algorithm**: Linear relationship between single feature (temperature) and target
- **Equation**: y = mx + b
- **Use Case**: Understanding basic relationships between variables
- **Pros**: Interpretable, fast, good starting point
- **Cons**: Only works for linear relationships

### 7. Multiple Linear Regression
- **Algorithm**: Linear relationship with multiple features
- **Features Used**: Temperature, humidity, dew point, radiation, rain, snow
- **Equation**: y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
- **Pros**: Captures multiple relationships, interpretable coefficients
- **Cons**: Assumes linearity, sensitive to outliers and multicollinearity

### 8. Neural Network Regression
- **Architecture**: 
  - Normalization layer for automatic input scaling
  - Single dense layer for regression output
  
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.1
- **Epochs**: 1000
- **Pros**: Handles non-linear relationships, learns complex patterns
- **Cons**: Requires more tuning and data, prone to overfitting

---

## Data Processing Pipeline

### 1. Data Loading and Cleaning
- Load datasets from CSV files
- Convert categorical variables to binary/numeric format
- Remove unnecessary columns

### 2. Scaling and Normalization
- **StandardScaler**: Normalizes features to zero mean and unit variance
  - Ensures all features contribute equally to model
  - Improves convergence speed for gradient-based algorithms
  
### 3. Handling Imbalanced Data
- **RandomOverSampler**: Resamples minority class to match majority class
  - Prevents models from biasing toward majority class
  - Important for fair model evaluation

### 4. Train-Validation-Test Split
- **Training Set**: 60% - Used to train the model
- **Validation Set**: 20% - Used during training to tune hyperparameters
- **Test Set**: 20% - Used for final evaluation on unseen data

---

## Key Concepts

### Loss Functions
- **Binary Crossentropy**: Used for classification
  - Measures divergence between predicted and actual probability distributions
  
- **Mean Squared Error (MSE)**: Used for regression
  - Average squared difference between predictions and actual values

### Regularization Techniques
- **Dropout**: Randomly disables neurons during training
  - Prevents co-adaptation of neurons
  - Acts as ensemble learning method
  
- **Early Stopping**: Monitors validation loss
  - Stops training when validation loss stops improving
  - Prevents overfitting

### Hyperparameter Tuning
- Grid search through combinations to find best performing model
- Evaluates validation loss on unseen validation data
- Selects model with lowest validation loss

---

## Model Comparison

| Model | Type | Training Speed | Interpretability | Handles Non-linearity |
|-------|------|-----------------|------------------|-----------------------|
| KNN | Instance-based | Slow | High | Yes |
| Naive Bayes | Probabilistic | Fast | High | No |
| Logistic Regression | Linear | Fast | High | No |
| SVM | Kernel-based | Moderate | Low | Yes |
| Neural Network | Deep Learning | Slow-Moderate | Low | Yes |

---

## Evaluation Metrics

### Classification
- **Precision**: Of predicted positives, how many are actually positive?
- **Recall**: Of actual positives, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

### Regression
- **R² Score**: Proportion of variance explained by the model
- **MSE**: Mean squared error between predictions and actual values
- **MAE**: Mean absolute error between predictions and actual values

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
imbalanced-learn (imblearn)
tensorflow
seaborn
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow seaborn
```

---

## Usage

1. Ensure required datasets are in the notebook directory:
   - `magic04.data` - Magic Gamma Telescope dataset
   - `SeoulBikeData.csv` - Seoul bike rental dataset

2. Run the notebook cells in order from top to bottom

3. Observe classification reports, plots, and model evaluation metrics

---

## Limitations

- Models require specific input shapes and feature counts
- Hyperparameter tuning can be computationally expensive
- Neural network performance depends heavily on data quality and quantity
- Results are specific to the datasets used
- No feature engineering is performed (advanced technique)
- No cross-validation is used (would provide more robust evaluation)

---

## Learning Outcomes

After completing this notebook, you should understand:
- How different ML algorithms work and their trade-offs
- When to use classification vs regression
- Importance of data preprocessing and scaling
- How to evaluate model performance
- Basics of neural network architecture
- Hyperparameter tuning and model selection

---

## Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [freeCodeCamp ML Course](https://www.freecodecamp.org/learn/machine-learning-with-python/)
- Original lecture video by [Kylie Ying](https://youtu.be/i_LwzRVP7bg?list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s)

Check out more amazing tutorials from [FreeCodeCamp](https://www.youtube.com/@freecodecamp)!