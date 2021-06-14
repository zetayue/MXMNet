# Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures

Code for the Multiplex Molecular Graph Neural Network (MXMNet) proposed in our [paper](https://arxiv.org/abs/2011.07457), which has been accepted by the Machine Learning for Structural Biology Workshop ([MLSB 2020](https://www.mlsb.io/)) and the Machine Learning for Molecules Workshop ([ML4Molecules 2020](https://ml4molecules.github.io/)) at the 34th Conference on Neural Information Processing Systems (NeurIPS 2020).

## Overall Architecture

<p align="center">
<img src="https://github.com/zetayue/MXMNet/blob/master/MXMNet.png?raw=true">
</p>

## Requirements
CUDA : 10.1
Python : 3.7.10

The other dependencies can be installed with:
```
pip install -r requirements.txt
```
## How to Run
You can directly download, preprocess the QM9 dataset and train the model with 
```
python main.py
```
Optional arguments:
```
  --gpu             GPU number
  --seed            random seed
  --epochs          number of epochs to train
  --lr              initial learning rate
  --wd              weight decay value
  --n_layer         number of hidden layers
  --dim             size of input hidden units
  --batch_size      batch size
  --target          index of target (0~11) for prediction on QM9
  --cutoff          distance cutoff used in the global layer
```
The default model to be trained is the MXMNet (BS=128, d_g=5) by using '--batch_size=128 --cutoff=5.0'.

## Cite
If you find this model and code are useful in your work, please cite our paper:
```
@inproceedings{zhang2020molecular,
  title     = {Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures},
  author    = {Zhang, Shuo and Liu, Yang and Xie, Lei},
  booktitle = {NeurIPS-W},
  year      = {2020}
}
```
