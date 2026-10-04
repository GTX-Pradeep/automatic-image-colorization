# Automatic Image Colorization

An ML-based project for automatically adding plausible colors to grayscale landscape images.

## Project Overview

This project explores automatic image colorization using luminance-chrominance color representation, superpixel segmentation, local image features, Support Vector Regression (SVR), and Markov Random Field (MRF) based smoothing.

## Pipeline

Grayscale Image  
→ YUV Conversion  
→ SLIC Superpixel Segmentation  
→ Local FFT Features  
→ SVR Color Prediction  
→ MRF / ICM Smoothing  
→ RGB Colorized Image

## Project Status

🚧 In development

## Dataset

Landscape images from the Kaggle Landscape Pictures dataset.

## Reference

The implementation is inspired by the provided automatic image colorization research paper and is being independently implemented and evaluated as part of an ML mini-project.
