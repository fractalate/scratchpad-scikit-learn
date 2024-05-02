# Wine Classification

Classify the wines as red or white based on the dataset available from [https://huggingface.co/datasets/mstz/wine](https://huggingface.co/datasets/mstz/wine).

## Setup

The following libraries are required to run the code from this project:

```
pip install numpy pandas scikit-learn jupyter matplotlib
```

See [requirements.txt](./requirements.txt) for the full list of dependencies.

## Usage

* The notebook [data_exploration.ipynp](./data_exploration.ipynp) can be used to investigate the dataset.
* To train and assess the model, run `python3 process.py`.
* To clear the training data, delete the `results` directory.
