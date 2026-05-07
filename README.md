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

## Try the Extension:
Try the extension:
1. Find the ai-detector-extension
2. Click the green Code button → Download ZIP
3. Unzip, then open chrome://extensions
4. Enable Developer mode → Load unpacked → select the ai-detector-extension folder
5. Pint it from the extensions menu

Using the Extension:
1. Click the extension icon --> the side panel willopen
2. Click "Scan Page"
3. The results will appear live as each of the images finishes, with a confidence bar


## References
- [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [Course Lab 6: PyTorch Doodle Classifier](https://colab.research.google.com/drive/1V1GsAgSpJ0y6FtE2x-mTpgLy3QASZuKE#scrollTo=RlAl_f_86Dsh)
- [PyTorch Official Tutorial](https://docs.pytorch.org)
- Metz, C., & Satariano, A. (2026, January 5). Nicolas Maduro AI images deepfakes. The New York Times. (https://www.nytimes.com/2026/01/05/technology/nicolas-maduro-ai-images-deepfakes.html)
- The Guardian. (2026, January 22). White House ICE protest arrest altered image. The Guardian. (https://www.theguardian.com/us-news/2026/jan/22/white-house-ice-protest-arrest-altered-image)
- Poynter. (2025). Trump White House AI political messaging. Poynter Institute. (https://www.poynter.org/fact-checking/2025/trump-white-house-ai-political-messaging/)
- Pew Research Center. (2024). Social media and news fact sheet. Pew Research Center. (https://www.pewresearch.org/journalism/fact-sheet/social-media-and-news-fact-sheet/)
- Velasquez-Salamanca, J., Martin-Pascual, M., & Andreu-Sanchez, C. (2025). Interpretation of AI-generated vs. human-made images. PMC. (https://pmc.ncbi.nlm.nih.gov/articles/PMC12295870/)
- Meta. (2024, February). Labeling AI-generated images on Facebook, Instagram, and Threads. Meta Newsroom. (https://about.fb.com/news/2024/02/labeling-ai-generated-images-on-facebook-instagram-and-threads/)
- Brookings Institution. (2023). Detecting AI fingerprints: A guide to watermarking and beyond. Brookings. (https://www.brookings.edu/articles/detecting-ai-fingerprints-a-guide-to-watermarking-and-beyond/)
- Mahara, T., et al. (2026, May). A broad overview of AI-generated image detection methods and their limitations. [Overview paper].
- Wang, S. Y., et al. (2020). CNN-generated images are surprisingly easy to spot... for now. CVPR.

