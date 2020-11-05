# Speech emotion recognition  PyTorch
This program is an implementation of 

[3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition](https://ieeexplore.ieee.org/document/8421023)

For tensorflow implimentation check out [author's repository](https://github.com/xuanjihe/speech-emotion-recognition)

# prerequesties

- Python 3.7 
  - if you want to use other version, just edit Pipfile and you're good to go
  
- Pipenv 
  - used for package mangement

# Installation
```
cd directory_of_this_repository
pipenv --python 3
```

# Training
## download dataset
You need to download IEMOCAP corpus as training data.
Labels for each sentences are given in `filelists/iemocap.csv`. I have split them in to train/val and saved in `filelists/iemocap_train.csv` `filelists/iemocap_val.csv`

## run training
```
pipenv shell # activate virtual environment
python train.py
```
Training logs are stored in `lightning_logs`

