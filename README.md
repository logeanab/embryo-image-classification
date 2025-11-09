## Embryo Classification Using Pre-trained CNN Models
## 🧫 Project Overview
This project focuses on improving embryo implantation success rates for an IVF and ART clinic. Currently, embryo grading is manual, subjective, and time-consuming. The goal is to develop a data-driven, AI-powered solution using deep learning to classify embryo images objectively and enhance fertility outcomes.

Business Problem: Manual embryo grading leads to inconsistent results and inefficiencies in the ART process. Solution: Implement Convolutional Neural Networks (CNNs) (MobileNet, EfficientNet, ResNet50) to automatically classify embryo images, reduce human bias, and improve decision accuracy.

## Objectives:
* Increase the success rate of ART procedures
* Automate embryo quality assessment using AI

## Success Criteria:
* Business: Improve ART success rate by at least 10%
* Economic: Achieve 25% cost savings by minimizing work on low-quality embryos
* Constraint: Reduce overall treatment cost while maintaining accuracy

## 🧠 Tech Stack
* Language: Python 3
* Deep Learning Framework: TensorFlow / Keras
* Pre-trained Models: MobileNet, EfficientNet, ResNet50
* Libraries: NumPy, Pandas, PIL (Pillow), scikit-learn
* Environment: Jupyter Notebook / Kaggle Notebook
* Data: Embryo images (train/test), CSV files for input/output

## 🏃 How to Run
1. Clone the Repository

git clone https://github.com/yourusername/embryo-classification.git
cd embryo-classification

2. Install Dependencies

pip install tensorflow pandas numpy pillow scikit-learn

3. Prepare Your Dataset

* Place your training and test images in the same folder structure as in the code:
  
/train/  → training images organized by label

/test/   → test images organized by label

* Ensure pre-trained models (MobileNet, EfficientNet, ResNet50) are in the /models/ folder or update paths in the .py files.

4. Run the Python Scripts

python MobileNet.py      # MobileNet model

python EfficientNet.py   # EfficientNet model

python ResNet50.py       # ResNet50 model

5. Check Results
* Predictions will be saved as CSV files (e.g., MobileNet.csv, EfficientNet.csv, ResNet50_submission.csv)

* Classification reports for train and test datasets will print in the console
