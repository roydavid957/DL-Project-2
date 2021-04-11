# DL-Project-2
Repo for the Deep Learning Project 2, Group 10

## How to RUN (RNN models):
- Place the 'data' folder within the RNN folder
- Running an individual experiment:
Open the terminal within RNN/ and specify the desired configuration with the arguments:
    * python3 run.py -M train -E MogLSTM -ED 2 -D MogLSTM -O ADAM -EN 100 -lr 0.001
    * outputs (within 'txt_results' folder):
        - bleu_train_GRU2GRU_d0.1_gc1.0_lr0.001.txt
        - loss_train_GRU2GRU_d0.1_gc1.0_lr0.001.txt

## How to visualize experiment results (RNN models):
- Open the terminal within RNN/ and run plot.py accompanied by the following arguments:
    * the datafile,
    * and the metric with the modified (relative to default settings) hyperparameters in the experiment
- Example: python3 plot.py -p "txt_results/GRU_experiments/bleu_train_GRU2GRU_d0.1_gc1.0_lr0.001.txt" -n "bleu_gc1.0"
- output: GRU2GRU_bleu_gc1.0.png

## How to RUN (BERT):
- Download the training data : formatted_movie_lines_train.txt
- Download the supporting scripts : 
     * finetune_on_pregenerated.py
     * pregenerate_training_data.py
     * simple_lm_finetuning.py
- Open the DL_BERT_GEN.ipynb notebook and run all (entire script will take 3-5 hours on Google Colab)
- Outputs a BERT model fine-tuned on training data

## How to run (GPT-2)
- Downloading the training datasets:
      * formatted_movie_lines_train.txt
      * formatted_movie_lines_QR_train.txt
- Download the Python scripts in the GPT2 folder
- (optional) Download the .sh files if you want to run them on Peregrine
- Run the Python script or the Shell script

## Qualitative analysis:
![alt text](https://github.com/roydavid957/DL-Project-2/blob/main/qas.png)
