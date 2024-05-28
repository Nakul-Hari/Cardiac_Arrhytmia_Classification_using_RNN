# Cardiac Arrhythmia Classification using RNN

## Overview
This repository contains code for classifying cardiac arrhythmias using Recurrent Neural Network (RNN) architectures, specifically Long Short-Term Memory (LSTM) and Bidirectional LSTM (BLSTM), trained on raw photoplethysmogram (PPG) signals. 

## Abstract
Cardiovascular diseases (CVDs) are a leading global cause of death, necessitating continuous monitoring for early detection. Traditional methods like electrocardiograms (ECGs) have limitations, leading to a shift towards using PPG signals due to their non-invasive nature. This project utilizes PPG signals to classify cardiac arrhythmias, employing signal preprocessing techniques and extracting comprehensive features. Classification is performed using LSTM and BLSTM RNN algorithms.

## Objectives
The main objective is to apply RNN techniques, specifically LSTM and BLSTM, to train a model to recognize different types of cardiac arrhythmias.

## Motivation
Heart diseases, particularly cardiac arrhythmias, pose significant health risks globally. Early detection and monitoring are crucial for preventing adverse events like heart attacks. While ECGs are commonly used, their limitations have led to exploring alternative methods like PPG signals.

## Methodology
### Dataset Extraction
PPG waveforms from the PhysioNet/Computing in Cardiology Challenge 2015 database were used for training and validation.
### Signal Pre-Processing
Signal pre-processing involved filtration, smoothing, baseline wandering removal, and normalization.
### Feature Extraction
Features including morphological, statistical, and frequency domain features were extracted from the PPG signals.
### Model Architecture
Two RNN-based models were employed: LSTM-based and Bidirectional LSTM-based architectures.
### Model Training
The models were trained and evaluated using both untrained and trained datasets, with evaluation metrics including loss and accuracy.

## Results and Discussion
The models demonstrated promising performance, with high accuracy and specificity. However, there is room for improvement, particularly in sensitivity. Experimentation with advanced neural network architectures and alternative preprocessing techniques could enhance performance.

## Conclusion and Future Perspective
Further refinement of the models, including feature set expansion and training with larger datasets, holds promise for improving performance. Adjusting classification thresholds and exploring specialized feature extraction methods are potential avenues for enhancement.

## References
- [Photoplethysmography Based Arrhythmia Detection and Classification](https://ieeexplore.ieee.org/document/8684801)
- [Cardiac Arrhythmias Classification Using Photoplethysmography Database](https://pubmed.ncbi.nlm.nih.gov/10858029/)
- [Cardiac Arrhythmia Detection Using Photoplethysmography](https://ieeexplore.ieee.org/document/7953658)
- [PhysioNet Challenge 2015 Dataset](https://archive.physionet.org/physiobank/database/challenge/2015/training/RECORDS)
