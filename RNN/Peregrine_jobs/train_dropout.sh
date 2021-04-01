#!/bin/bash

python run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.1
python run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.2
python run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.3

python run.py -M train -E LSTM -ED 2 -D LSTM -O ADAM -EN 50 -d 0.1
python run.py -M train -E LSTM -ED 2 -D LSTM -O ADAM -EN 50 -d 0.2
python run.py -M train -E LSTM -ED 2 -D LSTM -O ADAM -EN 50 -d 0.3