# Hieroglyph Sign Classifier

An end-to-end deep learning pipeline for detecting and classifying ancient Egyptian hieroglyphs from manuscript images.

**Live Demo:** https://juhi-hieroglyph-classifier.streamlit.app/

## Overview

This project fine-tunes a ResNet-50 model to classify hieroglyphs into 170 Gardiner sign categories, achieving **91.8% validation accuracy** on clean sign images. It also includes a full papyrus pipeline that detects individual signs from a manuscript column image using OpenCV contour analysis and classifies each one.

## Features

- **Single Sign Mode** - upload one hieroglyph image, get the Gardiner code + confidence score + top 5 predictions
- **Papyrus Column Mode** - upload a full column from a papyrus manuscript, auto-detect individual signs, classify each one, and list results top to bottom

## Model

| Detail | Value |
|---|---|
| Architecture | ResNet-50 (pretrained on ImageNet, fine-tuned) |
| Training images | 3,584 |
| Gardiner classes | 170 |
| Validation accuracy | 91.8% |
| Training epochs | 15 |
| Device | GPU (Google Colab T4) |

## Dataset

Based on the GlyphReader corpus (Franken & van Gemert, 2013), as used in Barucci et al. (2021). Source: `HamdiJr/Egyptian_hieroglyphs` on HuggingFace, manually annotated hieroglyph images labeled according to the Gardiner Sign List.

## Detection Pipeline

For manuscript images, individual signs are segmented using:
1. CLAHE contrast enhancement (handles aged papyrus coloring)
2. Otsu's adaptive thresholding
3. OpenCV contour detection with area filtering
4. Upscaling (6x) before detection for small/narrow images

This approach is consistent with the segmentation method used in Barucci et al. (2021).

## Limitations

- Classifier was trained on clean, isolated sign images. Confidence is lower on real manuscript images due to the domain gap between training data and aged papyri.
- Covers 170 of 750+ total Gardiner signs
- Detection works best on clear, high-contrast manuscript columns

## Research Context

This project is part of broader work on computational Egyptology, including NER on ancient Egyptian demonological texts (DemonThings/DemonBase) and Arabic NLP pipelines for archival documents. The domain gap between clean training images and real papyri represents the core open research problem. Fine-tuning on annotated manuscript data (e.g., PapyrusVision, OCR-PT-CT) is the proposed next step.

## Stack

- PyTorch + torchvision (ResNet-50)
- OpenCV (sign detection)
- Streamlit (demo interface)
- HuggingFace Datasets

## References

- Barucci et al. (2021). *A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification.* IEEE Access.
- Franken & van Gemert (2013). *Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts.* ACM MM.
- Gardiner, A. (1957). *Egyptian Grammar.* Griffith Institute.
