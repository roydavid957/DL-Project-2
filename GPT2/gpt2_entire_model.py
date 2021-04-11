# The following code loads the movie dialogue dataset,
# puts it into dataloaders, tokenizes them, and trains them
# with GPT-2.

# The data uses a formatted version of the dataset that was made
# outside of this script. It requires this specific dataset to run.
# The splits should be in the same github repository.

# -----------------------
# Imports
# -----------------------
import re, os, csv, unicodedata, codecs, itertools
import requests, random, time, datetime, pickle
import pandas as pd 
import numpy as np 
from io import open 
from itertools import compress
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW 
from nltk.translate.bleu_score import sentence_bleu

# -----------------------
# Loading the dataset
# -----------------------

# Load lines for training and BLEU scoring
train_lines = [line for line in open("formatted_movie_lines_train.txt", "r")]

# Instantiate the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    bos_token = "<|startoftext|>",
    eos_token = "<|endoftext|>",
    pad_token = "<|pad|>"
)

max_length = max([len(tokenizer.encode(line)) for line in train_lines])

# Class that makes the dataset
class DatasetMaker(Dataset):
    
    def __init__(self, txt_list, tokenizer, gpt2_type = "gpt2", maxlength = max_length):
      self.tokenizer = tokenizer # Instantiated outside
      self.input_ids = []
      self.attn_masks = []

      for txt in txt_list:
          encodings_dict = tokenizer(
              "<|startoftext|>" + txt + "<|endoftext|>",
              truncation = True,
              max_length = max_length,
              padding = "max_length"
          )

          self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
          self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__ (self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# Create the dataset
train_dataset = DatasetMaker(train_lines, tokenizer)

# Set the batch size
batch_size = 4

# Load the dataloader
train_dataloader = DataLoader(train_dataset, batch_size = batch_size)

# -----------------------
# GPT-2 Settings
# -----------------------

# Set the model configurations to the standard settings
config = GPT2Config.from_pretrained("gpt2", output_hidden_states = False)

# Create the instance of the model and set the token size embedding length
model = GPT2LMHeadModel.from_pretrained("gpt2", config = config)
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU
device = torch.device("cuda")
model.cuda()

# Variables for training parameters
epochs = 4
sample_every = 100

# Instantiate the optimizer, AdamW
optimizer = AdamW(model.parameters(), lr = 5e-4, eps = 1e-8)

# Count the total steps: data points * epochs
total_steps = len(train_dataloader) * epochs

# -----------------------
# Training the model
# -----------------------
def format_time(elapsed):
  return str(datetime.timedelta(seconds = int(round((elapsed)))))

# Set beginning of time when model runs
total_t0 = time.time()

# List that holds the information of the training stats
avg_training_loss = []
avg_bleu_scores = []

# Tell model to go to the device
model = model.to(device)

for epoch in range(0, epochs):
  
  print("----------------------------------------")
  print(f"Beginning epoch {epoch + 1} of {epochs}")
  t0 = time.time()
  total_train_loss = 0
  model.train()
  epoch_bleus = []

  # Gradient descent through batches
  for step, batch in enumerate(train_dataloader):
    b_input_ids = batch[0].to(device)
    b_labels = batch[0].to(device)
    b_masks = batch[1].to(device)

    model.zero_grad()

    outputs = model(
        b_input_ids,
        labels = b_labels,
        attention_mask = b_masks,
        token_type_ids = None
    )

    loss = outputs[0]

    batch_loss = loss.item()
    total_train_loss += batch_loss
    
    # Get sample of every 100 batches
    if step % sample_every == 0 and not step == 0:
      # Print the loss of the batch
      print(f"Batch {step} of {len(train_dataloader)}; Loss:{batch_loss}")

      # Calculate BLEU score
      model.eval()

      prompt = "<|startoftext|>"
      prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
      prompt = prompt.to(device)

      generated = model.generate(
        prompt,
        do_sample = True,
        top_k = 50,
        max_length = 40,
        top_p = 0.95,
        num_return_sequences = 1
      )

      candidate = tokenizer.decode(generated[0], skip_special_tokens = True)
      print(f"Example: {candidate}")

      model.train()

    loss.backward()
    optimizer.step()

  # Calculate the average loss over all batches
  avg_train_loss = total_train_loss / len(train_dataloader)
  avg_training_loss.append(avg_train_loss)

  # Measure how long this epoch took
  training_time = format_time(time.time() - t0)

  print(f"Average training loss: {avg_train_loss}; Epoch time: {training_time}")
  t0 = time.time()
  model.eval()
  total_eval_loss = 0
  nb_eval_steps = 0

print(f"Total training took {format_time(time.time() - total_t0)}")

# -----------------------
# Saving model and data
# -----------------------

# Save the model to generate new lines outside the script
model_path = "gpt2_entire_model"
model_to_save = model.module if hasattr(model, "module") else model
model_to_save.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save training losses
with open("gpt_entire_loss.txt", "w") as f:
  for item in avg_training_loss:
    f.write(f"{item}\n")
