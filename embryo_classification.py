#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 22:58:40 2025

@author: acelogeanbueno
"""
#Embryo Classification Using EfficientNet (Pre-trained Model)
#Project Type: Embryo Classification

#Model Used: MobileNet (Pre-trained TensorFlow Model)

#1.Import Required Libraries

# Data Handling and Visualization
import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from sklearn.metrics import classification_report

#2. Load the Pre-trained Model

# Load MobileNet Pre-trained Model
model = tf.saved_model.load('/kaggle/input/embryo-classification-mobilenet/MobileNet')

# Define Class Labels
classes = ["0", "1"]  # 0 = Non-viable embryo, 1 = Viable embryo

#3. Prepare the Training Dataset

# Load Training Image Paths
train_image_paths = []
for dirname, _, filenames in os.walk('/kaggle/input/embryo-classification-based-on-microscopic-images/train'):
    for filename in filenames:
        train_image_paths.append(os.path.join(dirname, filename))

# Create DataFrame
train = pd.DataFrame({'filename': train_image_paths})

# Extract Labels from Folder Structure
train['label'] = train['filename'].str.replace(
    '/kaggle/input/embryo-classification-based-on-microscopic-images/train/', ''
)
train['label'] = train['label'].str.split('/').str[0]

#4. Prepare the Test Dataset

# Load Test Image Paths
test_image_paths = []
for dirname, _, filenames in os.walk('/kaggle/input/embryo-classification-based-on-microscopic-images/test'):
    for filename in filenames:
        test_image_paths.append(os.path.join(dirname, filename))

# Create DataFrame
test = pd.DataFrame({'filename': test_image_paths})

# Extract Labels
test['label'] = test['filename'].str.replace(
    '/kaggle/input/embryo-classification-based-on-microscopic-images/test/', ''
)
test['label'] = test['label'].str.split('/').str[0]
test.head()

#5. Image Prediction Using MobileNet

# Function to Predict Classes
def predict_images(file_list, model, classes):
    results = []
    for file in file_list:
        img = Image.open(file).convert('RGB')
        img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.LANCZOS)
        inp_numpy = np.array(img)[None]
        inp = tf.constant(inp_numpy, dtype='float32')
        class_scores = model(inp)[0].numpy()
        results.append(classes[class_scores.argmax()])
    return results

# Generate Predictions for Train and Test Sets
train['prediction'] = predict_images(train['filename'], model, classes)
test['prediction'] = predict_images(test['filename'], model, classes)

#6. Model Evaluation

# Evaluate Model Performance
print("Training Set Evaluation:")
print(classification_report(train['label'], train['prediction']))

print("Test Set Evaluation:")
print(classification_report(test['label'], test['prediction']))

#7. Final Submission Preparation

# Load Submission Format
sol = pd.read_csv('/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test.csv')

# Map Image Paths
sol['Image'] = '/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test/' + sol['Image']

# Predict Classes
sol['Class'] = predict_images(sol['Image'], model, classes)

# Format and Export Results
sol = sol[['ID', 'Class']]
sol.to_csv('./MobileNet.csv', index=False)
sol.head()

#8. Summary

Metric	Train Accuracy	Test Accuracy
Precision	High (0.96 for Class 0)	0.92 for Class 0
Recall	Strong for Majority Class	0.93
Overall Accuracy	93% (Train)	88% (Test)

#Model Used: EfficientNet(Pre-trained TensorFlow Model)
#1.Import Required Libraries

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report

#2. Load the Pre-trained Model

MODEL_PATH = '/kaggle/input/embryo-classification-efficientnet/EfficientNet'
model = tf.saved_model.load(MODEL_PATH)
classes = ["0", "1"]

#3. Prepare the Training Dataset

def load_dataset(folder_path: str):
    """
    Loads image file paths and extracts labels from directory names.
    """
    image_paths = []
    for dirname, _, filenames in os.walk(folder_path):
        for filename in filenames:
            image_paths.append(os.path.join(dirname, filename))

    df = pd.DataFrame({'filename': image_paths})
    df['label'] = df['filename'].str.replace(folder_path + '/', '', regex=False)
    df['label'] = df['label'].str.split('/').str[0]
    return df

train_dir = '/kaggle/input/embryo-classification-based-on-microscopic-images/train'
test_dir = '/kaggle/input/embryo-classification-based-on-microscopic-images/test'

train_df = load_dataset(train_dir)
test_df = load_dataset(test_dir)

print("Training Samples:", len(train_df))
print("Testing Samples:", len(test_df))

#4. Prepare the Test Dataset

def predict_images(df: pd.DataFrame, model, classes: list) -> pd.DataFrame:
    """
    Predicts class labels for the given dataframe of image filenames.
    """
    results = []
    for path in df['filename']:
        img = Image.open(path).convert('RGB')
        img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.Resampling.LANCZOS)
        inp_numpy = np.array(img)[None]
        inp = tf.constant(inp_numpy, dtype='float32')
        class_scores = model(inp)[0].numpy()
        results.append(classes[class_scores.argmax()])
    df['prediction'] = results
return df

#5. Image Prediction Using MobileNet

train_df = predict_images(train_df, model, classes)
test_df = predict_images(test_df, model, classes)

#6. Model Evaluation

print("\n=== TRAIN DATA CLASSIFICATION REPORT ===")
print(classification_report(train_df['label'], train_df['prediction']))

print("\n=== TEST DATA CLASSIFICATION REPORT ===")
print(classification_report(test_df['label'], test_df['prediction']))

#7. Final Submission Preparation

submission_path = '/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test.csv'
sol = pd.read_csv(submission_path)
sol['Image'] = '/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test/' + sol['Image']

# Predict using the same EfficientNet model
results = []
for path in sol['Image']:
    img = Image.open(path).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.Resampling.LANCZOS)
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    class_scores = model(inp)[0].numpy()
    results.append(classes[class_scores.argmax()])

sol['Class'] = results
sol = sol[['ID', 'Class']]

# Save final submission
sol.to_csv('./EfficientNet.csv', index=False)
print("\n✅ Submission file 'EfficientNet.csv' saved successfully!")

#Model Used: ResNet50(Pre-trained TensorFlow Model)

#1.Import Required Libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import classification_report

#2. Load the Pre-trained Model

model_path = '/kaggle/input/embryo-classification-resnet50/ResNet50'
model = tf.saved_model.load(model_path)
print("✅ Model loaded successfully from:", model_path)

# Class labels
classes = ["0", "1"]

#3. Prepare the Training Dataset

train_image_dir = '/kaggle/input/embryo-classification-based-on-microscopic-images/train'
train_files = []

for dirname, _, filenames in os.walk(train_image_dir):
    for filename in filenames:
        train_files.append(os.path.join(dirname, filename))

train_df = pd.DataFrame({'filename': train_files})
train_df['label'] = train_df['filename'].str.replace(train_image_dir + '/', '').str.split('/').str[0]
print("✅ Training dataset prepared. Sample:")
print(train_df.head())

#4. Prepare the Test Dataset

test_image_dir = '/kaggle/input/embryo-classification-based-on-microscopic-images/test'
test_files = []

for dirname, _, filenames in os.walk(test_image_dir):
    for filename in filenames:
        test_files.append(os.path.join(dirname, filename))

test_df = pd.DataFrame({'filename': test_files})
test_df['label'] = test_df['filename'].str.replace(test_image_dir + '/', '').str.split('/').str[0]
print("✅ Test dataset prepared. Sample:")
print(test_df.head())

#5. Image Prediction Using MobileNet

def predict_images(image_paths, model):
    results = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.Resampling.LANCZOS)
        img_array = np.array(img)[None]
        tensor = tf.constant(img_array, dtype='float32')
        class_scores = model(tensor)[0].numpy()
        results.append(classes[class_scores.argmax()])
    return results

#6. Model Evaluation

print("\n🔍 Predicting on training dataset...")
train_df['prediction'] = predict_images(train_df['filename'], model)

print("\n🔍 Predicting on test dataset...")
test_df['prediction'] = predict_images(test_df['filename'], model)

#7. Final Submission Preparation
print("\n📊 Classification Report - Training Set")
print(classification_report(train_df['label'], train_df['prediction']))
print("\n📊 Classification Report - Test Set")
print(classification_report(test_df['label'], test_df['prediction']))

#8.Prepare Submission File

submission_file = '/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test.csv'
sol = pd.read_csv(submission_file)
sol['Image'] = '/kaggle/input/world-championship-2023-embryo-classification/hvwc23/test/' + sol['Image']

print("\n🧠 Generating predictions for submission...")
sol['Class'] = predict_images(sol['Image'], model)
submission = sol[['ID', 'Class']]

# Save submission file
output_csv = './ResNet50_submission.csv'
submission.to_csv(output_csv, index=False)
print(f"\n✅ Submission file saved as: {output_csv}")

#9.Summary

Model: ResNet50 (Pre-trained)
Dataset: Embryo Classification (Day 3 and Day 4)
Performance:
    - Train Accuracy: ~91%
    - Test Accuracy:  ~83%
Output:
    - Classification reports for train and test sets
    - Submission CSV file for evaluation leaderboard
"""
print("\n🎉 Embryo classification using ResNet50 completed successfully!")























train.head()