# Semantic segmentation in PyTorch using U-Net
<!-- ----------------------------------------------- -->

This is a simple example of implementing semantic segmentain using PyTorch.

## Download data
Download the data using 
```bash
$ bash data_scripts/download_data.sh
```
If it requires, enter your username and API key from [Kaggle](https://www.kaggle.com/docs/api#authentication) to download the data.
You should also accept the terms and conditions of [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge) on Kaggle.

## Train
To train the network, use
```bash
$ python train.py --data_dir <data_dir>
```
Currently, all parameters for training are provided in `config.py`. I should make these parameters available for user to set.

## Train on Google Colab
To train the network on Google Colab, please use the provided [jupyter notebook](https://github.com/AmirMardan/deep-learning-algorithms/blob/main/0_img_segmentation/1_unet_carvana/run_colab.ipynb).


