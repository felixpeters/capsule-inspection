# Capsule inspection

[![CI](https://github.com/felixpeters/capsule-inspection/actions/workflows/test.yaml/badge.svg)](https://github.com/felixpeters/capsule-inspection/actions/workflows/test.yaml)

Visual inspection for the pharmaceutical industry using convolutional neural networks.

## Proof of concept

### Classification models

Softgel classification model:

- Model: ResNet18
- Val. accuracy: 88.23%
- Val. AUC: 0.9122

Capsule classification model:

- Model: ResNet18
- Val. accuracy: 95.43%
- Val. AUC: 0.8858

### Segmentation models

Trained UNet models with a ResNet18 backbone which achieved very low loss values on both datasets.

## Links

Data:

- [SensumSODF Dataset](https://www.sensum.eu/sensumsodf-dataset/)

Tools:

- [Weights & Biases Dashboard](https://wandb.ai/felixpeters/capsule-inspection/overview)
