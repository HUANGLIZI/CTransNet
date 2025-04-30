This repo is the official implementation of **"Multimodal Multitask Deep Learning for Predicting Tertiary Lymphoid Structures and Peritoneal Recurrence in Gastric Cancer: A Multicenter Study"**.

<!-- ![image](https://github.com/HUANGLIZI/CTransNet/blob/main/CTransNet.png) -->

## Requirements

Python == 3.8 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage

### 1. Training

You can train to get your own model.

```angular2html
python train_vit.py
```

### 2. Evaluation

#### 2.1. Test the Model on the prediction of the maturation stages of TLSs.

```angular2html
python eval.py
```
#### 2.2. Test the Model on the prediction of peritoneal recurrence-free survival

```angular2html
python eval_os.py
```
