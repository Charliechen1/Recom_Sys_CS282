#!/bin/bash

# script to download Amazon review data
# e.g. $1 = Gift_Cards for gift_card domain data
#
# for more details please refer to http://deepyeti.ucsd.edu/jianmo/amazon/index.html
# right click on the hyper link, choose copy url to see the name of the dataset
# Generally speaking, replace the space with underscore works.
# e.g. Gift Cards -> Gift_Cards

echo "downloading $1 data"
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/$1.json.gz

echo "downloading $1 meta"
wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/$1.json.gz
