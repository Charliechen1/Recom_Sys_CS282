#!/bin/bash

# script to download Amazon review data
# $1 = Gift_Cards for gift_card domain data
echo "downloading $1 data"
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/$1.json.gz
