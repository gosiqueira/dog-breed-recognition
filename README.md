# Dog Breed Recognition

## Description

Develop a simple dog breed recognition system. However, with an extra difficulty: adding races that weren't seen in training time, and knowing how to identify unknowns (for example, our dear mutts).

## Requirements

To proper run this repo you need to have installed the following packages:
- Python 3.6 or superior
- Matplotlib
- Numpy
- Scikit-Learn
- PyTorch
- Torchvision

## Dataset

Dog Breeds dataset contains image from 100 distinct ldog-breeds to train a model. Additionally, it provides a subset for `enroll` new classes to the trained model and a final `test` set to evaluate the `enroll`.

Download the dog breeds dataset [here](https://drive.google.com/file/d/1-njeI_NWA6_Gz0Bu-tlAduktyjebzUmq/view?usp=sharing).

## Pre-trained model

Download original 100 dog breeds recognition model weights available [here]().

Download enrolled xxx dog breeds recognition model weights available [here]().

## Instructions

There are basically 3 main operations to perform using this repo:
- train: train a dog breed recognition model from scratch
- enroll: replace the original dog breeds and enroll new ones to the model
- eval: evaluate the enrolled breeds

### Train

To re-train the model, run the following command:

```python
python train.py <dirpath> <outpath> --epochs <num_epochs> --learning-rate <learning_rate> --batch-size <batch_size>
```

- `dirpath`: the dataset directory path
- `outpath`: a path to store the training history and the model weights
- `epochs`: the number of training epochs
- `learning-rate`: optimizer learning rate value
- `batch-size`: the number of instances to compose a mini-batch

### Enroll

To enroll new dog breeds, run the following command:

```python
python enroll.py <dirpath> <outpath> <modelpath> --epochs <num_epochs> --learning-rate <learning_rate> --batch-size <batch_size>
```

- `dirpath`: the dataset directory path
- `outpath`: a path to store the enroll history and the enrolled model weights
- `modelpath`: a path to the pre-trained dog breed recognition model
- `epochs`: the number of fine-tuning epochs
- `learning-rate`: optimizer learning rate value
- `batch-size`: the number of instances to compose a mini-batch

### Eval

To evaluate the enrolled breeds, run the following command:

```python
python eval.py <dirpath> <outpath> <modelpath> --batch-size <batch_size>
```

- `dirpath`: the dataset directory path
- `outpath`: a path to store the evaluation history
- `modelpath`: a path to the pre-trained dog breed recognition model
- `batch-size`: the number of instances to compose a mini-batch

## Contributions

I'm always opened to discussions and contributions. So, if you find interesting what I developed here and have any suggestion to enhance the codes, please, let me now in the [issues](https://github.com/gosiqueira/dog-breed-recognition/issues).
