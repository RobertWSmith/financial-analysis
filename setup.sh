#!/usr/bin/env bash

conda env create --file environment.yaml --force -y

cd python
# slap a `-e` in there to play with it
pip install .
# now, you can run the notebook
