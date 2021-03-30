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
import requests, random, time, datetime
import pandas as pd 
import numpy as np 
from io import open 
from itertools import compress
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

# -----------------------
# Loading the dataset
# -----------------------
train_lines_path = "formatted_movie_lines_train.txt"
validation_lines_path = "formatted_movie_lines_valid.txt"

train_lines = []
valid_lines = []

with open(train_lines_path, "r", encoding = "Windows-1252") as f:
    for line in f:
        train_lines.append(line)

with open(validation_lines_path, "r", encoding = "Windows-1252") as f:
    for line in f:
        valid_lines.append(line)

for i, line in enumerate(train_lines):
    new_line = re.sub(r"[\t\n]", r" ", line)
    new_line = re.sub(r"[Â]", r"", new_line)
    train_lines[i] = new_line

for i, line in enumerate(valid_lines):
    new_line = re.sub(r"[\t\n]", r" ", line)
    new_line = re.sub(r"[Â]", r"", new_line)
    valid_lines[i] = new_line

# Truncate the model because it cannot handle more than 1024 tokens
for i in range(len(train_lines)):
    line = train_lines[i]
    if len(line) > 1024:
        line = line[:1024]
        train_lines[i] = line

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

# Create the datasets
train_dataset = DatasetMaker(train_lines, tokenizer)
val_dataset = DatasetMaker(valid_lines, tokenizer)

# Set the batch size
bs = 4

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), # Sampling for training is random
            batch_size = bs
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), # Sampling for validation is sequential as the order doesn't matter.
            batch_size = bs 
        )

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
warmup_steps = 1e2
sample_every = 100

# Instantiate the optimizer
# Here, we will use AdamW
optimizer = AdamW(model.parameters(),
                  lr = 5e-4,
                  eps = 1e-8)

# Count the total steps: data points * epochs
total_steps = len(train_dataloader) * epochs

# Set a variable learning rate
# Scan large areas with high learning rate, 
# then finetune with lower learning rate
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)

# -----------------------
# Training the model
# -----------------------
def format_time(elapsed):
  return str(datetime.timedelta(seconds = int(round((elapsed)))))

# Set beginning of time when model runs
total_t0 = time.time()

# List that holds the information of the training stats
training_stats = []

# Tell model to go to the device
model = model.to(device)

for epoch in range(0, epochs):
  
  print(f"Beginning epoch {epoch + 1} of {epochs}")
  t0 = time.time()
  total_train_loss = 0
  model.train()

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
      elapsed = format_time(time.time() - t0)
      
      # Print the loss of the batch
      print(f"Batch {step} of {len(train_dataloader)}; Loss:{batch_loss}; Time:{elapsed}")

      model.eval()

      sample_outputs = model.generate(
          bos_token_id = random.randint(1, 30000),
          do_sample = True,
          top_k = 50,
          max_length = 200,
          top_p = 0.95,
          num_return_sequences = 1
      )

      for i, sample_output in enumerate(sample_outputs):
        print(f"Example output: {tokenizer.decode(sample_output, skip_special_tokens = True)}")
      
      model.train()

    loss.backward()
    optimizer.step()
    scheduler.step()

  # Calculate the average loss over all batches
  avg_train_loss = total_train_loss / len(train_dataloader)

  # Measure how long this epoch took
  training_time = format_time(time.time() - t0)

  print(f"Average training loss: {avg_train_loss}; Epoch time: {training_time}")
  t0 = time.time()
  model.eval()
  total_eval_loss = 0
  nb_eval_steps = 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_labels = batch[0].to(device)
    b_masks = batch[1].to(device)

    with torch.no_grad():
      outputs = model(
          b_input_ids,
          attention_mask = b_masks,
          labels = b_labels
      )
      loss = outputs[0]

    batch_loss = loss.item()
    total_eval_loss += batch_loss
  
  avg_val_loss = total_eval_loss / len(validation_dataloader)
  validation_time = format_time(time.time() - t0)
  print(f"Validation loss: {avg_val_loss}; Validation time: {validation_time}")

  # Record all statistics from this epoch
  training_stats.append(
      {
          "epoch": epoch + 1,
          "Training Loss": avg_train_loss,
          "Valid Loss": avg_val_loss,
          "Training Time": training_time,
          "Validation Time": validation_time
      }
  )

print(f"Total training took {format_time(time.time() - total_t0)}")