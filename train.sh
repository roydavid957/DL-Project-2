#!/bin/bash

python run.py -M train -E GRU -ED 2 -D GRU -O ADAM
python run.py -M train -E MogLSTM -ED 2 -D MogLSTM -O ADAM