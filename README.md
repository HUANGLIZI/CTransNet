This repo is the official implementation of **"Multimodal Multitask Deep Learning for Predicting Tertiary Lymphoid Structures and Peritoneal Recurrence in Gastric Cancer: A Multicenter Study"**.

![image](https://github.com/HUANGLIZI/CTransNet/blob/main/CTransNet.png)

## Requirements

Python == 3.8 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
## Pre-trained model

The model weights of CTransNet can be downloaded at the [link](https://drive.google.com/file/d/1LKFHLwJS8N0gJ8fkH4U-TI06tLMmzNZb/view?usp=sharing). You may need to send the request to get the access permission.

## Usage

### 1. Training

You can train to get your own model.

```angular2html
python train.py
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
#### 2.3. Inference the Model to predict the maturation stages of TLSs and peritoneal recurrence-free survival on external validation cohort

```angular2html
python inference.py
```
