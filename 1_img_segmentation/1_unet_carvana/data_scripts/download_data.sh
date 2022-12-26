#!/bin/bash

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -n "Kaggle username: "
    read USERNAME
    echo 
    echo -n "Kaggle API key: "
    read APIKEY

    mkdir -p ~/.kaggle
    echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi

pip install --upgrade kaggle


kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip

unzip train_hq.zip
unzip train_masks.zip
    
mkdir -p data/carvana/imgs
mv train_hq/* data/carvana/imgs/
rm -r train_hq
rm train_hq.zip

mkdir -p data/carvana/masks 
mv train_masks/* data/carvana/masks/
rm -d train_masks
rm train_masks.zip

echo "Images are downloaded and saved in ../data/"