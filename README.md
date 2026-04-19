Chest X-Ray Classification: Cardiomegaly Detection
A deep learning pipeline for binary classification of chest X-rays, trained to detect Cardiomegaly (enlarged heart) using transfer learning with InceptionV3. Built and run on Google Colab with GPU acceleration.

Overview
This notebook fine-tunes a pre-trained InceptionV3 model on a curated subset of chest X-ray images to classify whether a given scan shows signs of Cardiomegaly or not. It is based on the medical-ai repository, which provides labelled chest X-ray data and helper utilities.

Dataset

Source: medical-ai GitHub repository (labels.csv + image files)
Label format: CSV with columns — filename, height, width, label, xmin, ymin, xmax, ymax, view
Total labelled records: 1,964 across 9 pathology categories (e.g. Atelectasis, Cardiomegaly, Pneumothorax, Nodule, etc.)
Target condition: Cardiomegaly (positive class) vs. No Finding (negative class)
Split:

Training set: 232 images
Validation/Test set: 60 images


Image views: Posteroanterior (PA) and Anteroposterior (AP)

