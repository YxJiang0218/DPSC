# Adaptive Subspace Clustering via Diffusion Probabilistic Autoencoders

## Introduction
This repository contains the PyTorch implementation of the paper *Adaptive Subspace Clustering via Diffusion Probabilistic Autoencoders*.

Here, you will find code to execute clustering with representation learned by pre-trained and fine-tuned models.

## Dataset

This repository contains the data we used in experiments. For all the 5 datasets, we provide their semantic subcode produced by different pre-tained Diffusion Probabilistic Autoencoders as well as the corresponding labels.

## Code
To install dependencies, run
```
pip install -r requirements.txt
```
### Using pre-trained models (example)
For getting the results via pre-trained models, run
```bash
python main.py --dataset=coil20 --pretrained=bd --lr=1e-3
```
where you can choose 'dataset' from coil20/coil40/mnist/usps/orl, 'pretrained' from ffhq/bd/horse.

### Fine-tuned models
For getting the results via fine-tuned models and check the loss/accuracy/NMI during training, details can be found at [train_detail.ipynb](train_detail.ipynb)

## Contact
Yuxuan Jiang via yjiangei@connect.ust.hk