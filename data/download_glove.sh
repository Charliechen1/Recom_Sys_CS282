#!/bin/bash

# script to download Amazon review data
# e.g. $1 = Gift_Cards for gift_card domain data
#
# for more details please refer to http://deepyeti.ucsd.edu/jianmo/amazon/index.html
# right click on the hyper link, choose copy url to see the name of the dataset
# Generally speaking, replace the space with underscore works.
# e.g. Gift Cards -> Gift_Cards

echo "downloading GloVe 6B"
wget http://nlp.stanford.edu/data/glove.6B.zip

echo "unzipping GloVe 6B"
unzip glove.6B.zip -d glove_6B
