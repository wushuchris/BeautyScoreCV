# Facial Beauty Scoring Using Deep Learning

## Authors:
**Christopher Mendoza, Ian Lucas**  
**Date:** December 11, 2024  

---

## Overview
This project explores the prediction of facial beauty scores using deep learning techniques in computer vision. The goal is to develop a machine learning model capable of scoring facial images based on human-perceived beauty using the **SCUT-FBP5500 dataset**.

---

## Dataset
- **Name:** SCUT-FBP5500 Dataset (Available on Kaggle)  
- **Description:** A dataset of facial images labeled with beauty scores.  
- **Structure:**  
  - **Image Name:** Filename of the image (e.g., `img_1.jpg`).  
  - **Beauty Score:** A numerical value representing the beauty score (e.g., 4.2).  

---

## Project Workflow

### **1. Exploratory Data Analysis (EDA)**

#### Objectives:
- Understand the dataset structure and beauty score distribution.
- Visualize correlations between beauty scores and image characteristics (e.g., brightness, contrast).
- Inspect example images grouped by varying beauty scores.

#### Key Steps:
- **Beauty Score Distribution:** Analyze frequency and range of scores.
- **Correlation Analysis:** Determine if image properties influence beauty scores.
- **Sample Image Visualization:** Display images alongside their scores for qualitative insights.

---

### **2. Data Preprocessing**

#### Key Steps:
1. **Image Resizing:** All images were resized to `224x224` pixels.
2. **Normalization:** Pixel values were scaled to a range of `[0, 1]`.
3. **Data Augmentation:**
   - Applied transformations: rotation, flipping, shifting, and zooming.
   - Visualized augmented samples to ensure diversity in the training set.
4. **Dataset Splitting:**
   - Train (60%), Validation (20%), Test (20%) splits.
   - Ensured balanced beauty score distribution across splits.

#### Why This Matters:
- Resizing and normalization standardize input for the model.
- Augmentation improves model robustness and reduces overfitting.
- Splitting ensures unbiased evaluation and reproducibility.

---

### **3. Model Development**

#### Initial Attempts:
- **Feature-Enhanced Linear Regression:**
  - Features extracted from a pretrained VGG16 model.
  - Linear regression as the predictive layer.
- **Challenges:**
  - Linear regression couldn’t capture the non-linear nature of beauty perception.
  - Limited ability to optimize features for beauty scoring.

#### Final Approach:
- **Deep Learning Model:**
  - Fine-tuned VGG16 pretrained model for end-to-end learning.
  - Used Convolutional Neural Networks (CNNs) for feature extraction and regression.
  - Implemented a weighted loss function to handle score imbalances.

---

### **4. Results and Evaluation**

#### Metrics:
- **Mean Absolute Error (MAE):** Measures the average difference between predicted and actual scores.
- **Root Mean Squared Error (RMSE):** Highlights the magnitude of prediction errors.
- **R-squared (R²):** Indicates how well the model explains the variance in beauty scores.

#### Visualizations:
1. **Feature Maps:** Intermediate convolutional layers’ activations for a test image.
2. **Predictions vs. Actual Scores:** Scatter plots to assess model accuracy.
3. **Top 5 and Bottom 5 Images:** Comparison of the best and worst predictions.

---

### **5. Challenges and Future Work**

#### Challenges:
- Subjectivity in beauty scoring creates inherent noise in the data.
- Dataset imbalance in beauty scores affects prediction accuracy.

#### Future Directions:
- Train on a more diverse dataset to improve generalization.
- Explore ensemble models combining CNNs with handcrafted features (e.g., symmetry, texture).
- Apply transfer learning with recent architectures like ResNet or EfficientNet.

---

## Technical Details

### Environment:
- Python 3.8
- TensorFlow/Keras
- NumPy, Matplotlib
- scikit-learn

### Dependencies:

Install dependencies using:
```bash
pip install -r requirements.txt
