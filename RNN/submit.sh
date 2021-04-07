#!/bin/bash

sbatch train_GRU_clip.sh
sbatch train_GRU_drp.sh
sbatch train_GRU_lr.sh
sbatch train_LSTM_clip.sh
sbatch train_LSTM_drp.sh
sbatch train_LSTM_lr.sh
sbatch train_MogLSTM_clip.sh
sbatch train_MogLSTM_drp.sh
sbatch train_MogLSTM_lr.sh