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


kaggle datasets download -d kumaresanmanickavelu/lyft-udacity-challenge

unzip lyft-udacity-challenge.zip
    
mkdir -p data/lyft_udacity
mv dataA data/lyft_udacity/
mv dataB data/lyft_udacity/
mv dataC data/lyft_udacity/
mv dataD data/lyft_udacity/
mv dataE data/lyft_udacity/
# rm lyft-udacity-challenge.zip

echo "Images are downloaded and saved in ../data/lyft_udacity"