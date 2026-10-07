Medical Image Kernel and Dilation Analysis

Problem Statement
A medical-image system must detect both fine boundaries and broad structures. Different convolution kernel sizes and dilation settings affect the receptive field and the type of visual information captured from a medical image.
This project implements different kernel sizes and dilation settings and analyzes their receptive fields and feature-map responses using breast ultrasound images.

Objective
The main objectives are:
To analyze the effect of different convolution kernel sizes.
To analyze the effect of dilation on the receptive field.
To compare feature-map responses produced by different configurations.
To understand which settings are more suitable for fine details and broader structures.

Dataset
The project uses the BUSI-WHU Breast Ultrasound Dataset.
The dataset contains breast ultrasound medical images and corresponding annotations.
Due to the large number of image files, the complete dataset is not uploaded to this GitHub repository.
After downloading the dataset, place the required processed images inside:

dataset/
└── processed/
    ├── train/
    ├── val/
    └── test/
Convolution Configurations

Four configurations are analyzed:

Configuration	Kernel	Dilation	Effective Receptive Field
3×3 D=1	3×3	1	3×3
5×5 D=1	5×5	1	5×5
3×3 D=2	3×3	2	5×5
3×3 D=3	3×3	3	7×7

The effective receptive field is calculated using:

RF = (K - 1) × D + 1

where:

K = kernel size
D = dilation rate
RF = effective receptive field

Feature Map Analysis
A medical ultrasound image is passed through each convolution configuration.
The resulting feature maps are compared to understand how the kernel size and dilation affect the captured image information.
The feature-map comparison is saved as:
results/feature_map_comparison.png
Receptive Field Analysis

The receptive field sizes of the four configurations are compared using a graph.

The graph is saved as:

results/receptive_field_comparison.png
Interpretation
3×3, D=1: focuses on fine and local image details.
5×5, D=1: captures broader local structures.
3×3, D=2: provides a larger receptive field while keeping a small kernel.
3×3, D=3: provides the broadest receptive field among the tested configurations.
Conclusion

Increasing the kernel size or dilation increases the effective receptive field. Small kernels with low dilation are useful for detecting fine details and boundaries, while larger kernels or higher dilation help capture broader structures and contextual information.

Dilation is particularly useful because it increases the receptive field without increasing the actual kernel size.

Project Structure
medical_kernel/
│
├── dataset/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── feature_analysis.py
│   └── receptive_field_analysis.py
│
├── results/
│   ├── feature_map_comparison.png
│   └── receptive_field_comparison.png

Requirements
Python 3.11
PyTorch
Torchvision
NumPy
Pillow
Matplotlib
Scikit-learn

How to Run
Activate the Python virtual environment:
.\medicalenv\Scripts\Activate.ps1
Run feature-map analysis:
python src\feature_analysis.py
Run receptive-field analysis:
python src\receptive_field_analysis.py
The generated results will be stored in the results folder.
