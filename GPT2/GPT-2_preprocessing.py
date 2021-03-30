# -----------------------
# Imports
# -----------------------
import re, os, csv, unicodedata, codecs, itertools
from io import open

# -----------------------
# Preprocessing Functions
# -----------------------

# Function that splits each line of the file into a dictionary of fields
def load_lines(filename, fields):
    lines = {}
    with open(filename, "r", encoding = "iso-8859-1") as f:
        for line in f:

            # Split each line by this weird symbol
            values = line.split(" +++$+++ ")
            
            # Extract the fields
            # Each txt file have informative fields 
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
                


# -----------------------
# Stuff for colab
# -----------------------

class DatasetMaker(Dataset):
    
    def __init__ (self, txt_list, tokenizer, gpt2_type = "gpt2", max_length = max_length)

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