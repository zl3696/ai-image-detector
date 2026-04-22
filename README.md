# Truth in the Noise: Spotting AI Images in a Synthetic World

## Overview
With our interest in photojournalism and AI security, we aimed to address the real world concern about AI entering the field of journalism. This project uses machine learning approaches to detect AI-generated images at pixel level, trained on the CIFAKE dataset of 120,000 real and AI-generated images. The end goal is to write a technical blog post and develop a Chrome browser extension that automatically flags suspicious images on web.

## Dataset
[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 60,000 real images (from CIFAR-10)
- 60,000 AI-generated images
- Total: 120,000 labeled images

## Models
- Baseline CNN (trained from scratch)
- ResNet-18 (transfer learning, ImageNet pretrained)

## Results

| Model | Accuracy | FAKE Recall | REAL Recall |
|---|---|---|---|
| Baseline CNN | 93.67% | 0.9242 | 0.9492 |
| ResNet-18 Transfer | 88.58% | 0.9123 | 0.8592 |

The Baseline CNN outperforms ResNet-18 transfer learning on this task. See `docs/blog.md` for full analysis.

## Repo Structure
- `src/` — model, dataset, training, and evaluation code
- `pipeline.ipynb` — full reproducible experiment notebook
- `assets/` — figures from training and evaluation
- `docs/` — project blog and writeup
- `data/` — dataset information

## How to Reproduce
1. Open `pipeline.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells in order

## References
- [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [Course Lab 6: PyTorch Doodle Classifier](https://colab.research.google.com/drive/1V1GsAgSpJ0y6FtE2x-mTpgLy3QASZuKE#scrollTo=RlAl_f_86Dsh)
- [PyTorch Official Tutorial](https://docs.pytorch.org)
