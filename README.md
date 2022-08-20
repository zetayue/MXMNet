# Graph Neural Network for Predicting Molecular Properties
This project evaluates different Graph Neural Network(GNN) architectures for their effectiveness in predicting the Quantum Mechanical properties of chemical molecules.

## Requirements
We use PytorchLightning and PytorchGeometric as development frameworks and Weights & Biases for experiment management. 

## Setup
Install dependencies by running
 
```bash
sh setup.sh
```

## Run
To run DAGNN model use the following command

```bash
python main.py --target=7 --lr=0.0001 --n_layer=2 --dagnn=True
```

To run the baseline model

```bash
python main.py --target=7 --lr=0.0001
```

To run the model with Virtual Node

```bash
python main.py --target=7 --lr=0.0001 --n_layer=6 --virtual_node=True
```

To run the model with Auxiliary Layer

```bash
python main.py --target=7 --lr=0.0001 --n_layer=4 --auxiliary_layer=True
```

Optional arguments:
```python
  --wandb           Enable Weights & Biases to track experiment
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


## References
- The base MXMNet Implementation is taken from https://github.com/zetayue/MXMNet
- Auxiliary Layer implementation is inspired from https://github.com/rasbt/machine-learning-book/blob/main/ch18/ch18_part2.py
- Virtual Node and DAGNN Layer implementation is taken from https://github.com/divelab/MoleculeX/blob/master/BasicProp/kddcup2021/deeper_dagnn.py
