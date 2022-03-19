#!/usr/bin/env bash

conda env create --file environment.yaml --force -y

pip install .
# now, you can run the notebook
