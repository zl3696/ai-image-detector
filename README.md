# Truth in the Noise: Spotting AI Images in a Synthetic World

## Overview
With our interest in photojournalism and AI security, we aimed to address the real world concern about AI entering the field of journalism. This project uses machine learning approaches to detect AI-generated images at pixel level, trained on the CIFAKE dataset of 120,000 real and AI-generated images. The end goal is to write a technical blog post and develop a Chrome browser extension that automatically flags suspicious images on web.

## Dataset
[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 60,000 real images (from CIFAR-10)
- 60,000 AI-generated images
- Total: 120,000 labeled images

## Models
- Baseline CNN (from scratch)
- ResNet Transfer Learning (coming soon)

## Results
- Baseline CNN Accuracy: ~93%

## Pipeline
1. Data Loading & Preprocessing
2. Custom PyTorch Dataset & DataLoader
3. CNN Model Training
4. Evaluation

## References
- [Course Lab 6: PyTorch Doodle Classifier](https://colab.research.google.com/drive/1V1GsAgSpJ0y6FtE2x-mTpgLy3QASZuKE#scrollTo=RlAl_f_86Dsh)
- [PyTorch Official Tutorial](https://docs.pytorch.org)
