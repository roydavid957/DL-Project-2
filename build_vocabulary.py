import torch
from torch import LongTensor, Tensor, BoolTensor

import os
import itertools
import re
import unicodedata
import random
from typing import List, Tuple, Union

from format_data import corpus_name, datafile

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3  # Minimum word count threshold for trimming


class Voc:
    """ Vocabulary class:\n
        - Maps words to indices, indices to words.
        - Keeps track of count of each word, and total word count.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}  # number of times the word has occurred in the dataset
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Number of words in the dictionary, also counts in SOS, EOS, PAD, hence the assignment to 3

    def addSentence(self, sentence: str):
        """ Adds an entire sentence to the vocabulary by calling addWord on each word. """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str):
        """ Adds a word to the vocabulary transformed to an index, or increases the word's count in the vocabulary """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count: int):
        """
        Remove words below a certain count threshold from the entire vocabulary,
        and recreate the vocabulary with the remaining words.
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s: str):
    """ Lowercase, trim, and remove non-letter characters\n

     Examples:\n
     - Ew, it's like some gross rat...
     --> ew it s like some gross rat . . .

     - "I know... I still can't get over that his name was \"\"\"Seymour.\"\"\"
     --> i know . . . i still can t get over that his name was seymour ."""

    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def filterPair(p: List[str]) -> bool:
    """
    :param p: List of query-reply sentence pairs, processed by normalizeString() and readVocs()
            - Example: ['the money s right here ! get the key !', 'no ! you get it !']
    :return: True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs: List[List[str]]) -> List[List[str]]:
    """
    :param pairs: The entire dataset in query-reply sentence form.
    :return: Filtered query-reply dataset.
    """
    return [pair for pair in pairs if filterPair(pair)]


# Symbol Pre-processing step 1
def readVocs(datafile: str, corpus_name: str) -> Tuple[Voc, List[List[str]]]:
    """
    Read query/response pairs and return an empty Voc object with the pairs for loadPrepareData()\n
    :param datafile: Txt file containing the conversations, e.g. "formatted_movie_lines.txt".
    :param corpus_name: Name to be assigned to the instance of the Voc class.
    :return: Empty Voc object and normalized list of query-reply sentence pairs
            - Example pair from pairs: ['you d go with him !', 'don t kid yourself you know how i stand back there .']
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Symbol Pre-processing step 2
def loadPrepareData(corpus_name: str, datafile: str) -> Tuple[Voc, List[List[str]]]:
    """
    Using readVocs(), return a populated voc object and a query-reply pairs list of lists.
    :param corpus_name: Name of the Voc object.
    :param datafile: Path to formatted dataset.
    :return: Voc object and the dataset (pairs) before the last (trimming) symbol pre-processing step.
    """
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Symbol Pre-processing step 3
def trimRareWords(voc: Voc, pairs: List[List[str]], MIN_COUNT:int) -> List[List[str]]:
    """
    Trim words used under the MIN_COUNT from the voc and from pairs
    :param voc: Populated and untrimmed Voc object.
    :param pairs: Pairs that have been filtered according to their length.
    :param MIN_COUNT: The minimum number of times the word has to appear in the vocabulary
    :return: The trimmed pairs list. Voc object is trimmed inplace, doesn't need to be returned.
    """
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# Number pre-processing step 0
def indexesFromSentence(voc: Voc, sentence: str) -> List[int]:
    """
    Transform symbols to numbers. Preparing the data for embeddings.
    :param voc: Populated and trimmed Voc object.
    :param sentence: Either a query or a reply type of sentence.
    :return: A list of words from the sentence encoded as integer numbers.
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]  # EOS_token = 2


# Number pre-processing step 1
def zeroPadding(indexes_batch: List[List[int]], fillvalue=PAD_token) -> List[Tuple[int]]:  # PAD_token = 0
    """
    A function that takes a batch of word2idx encoded inputs, searches for the longest sequence in the batch and
    pads the rest of the batch to match the longest length.
    :param indexes_batch: Either query or reply batch of sentences.
            Example with a batch size of 2: [[16, 937, 4, 2], [42, 2]]
    :param fillvalue: The value used for padding.
    :return: The indexed batches in a list of tuples, where each tuple has one element from each sentence from the same
            position of the respective sentences.
            E.g. (72, 276, 115, 16, 42) would be the tuple of first word from 5 distinct sentences.
    """
    return list(itertools.zip_longest(*indexes_batch, fillvalue=fillvalue))


# Number pre-processing step 2 (for replies)
def binaryMatrix(replies_padlist: List[Tuple[int]]) -> List[List[bool]]:
    """
    A function that transforms a list of tuples with integers and transforms the integers to a binary representation.
    Used as a mask for batches.
    """
    bin_matrix = []
    for i, seq in enumerate(replies_padlist):
        bin_matrix.append([])
        for token in seq:
            if token == PAD_token:
                bin_matrix[i].append(0)
            else:
                bin_matrix[i].append(1)
    return bin_matrix


def inputVar(queries: List[str], voc: Voc) -> Tuple[LongTensor, Tensor]:
    """
    Returns padded input sequence tensor and their sentc_lengths.
    :param queries: A batch of query sentences. Default batch_size in train.py is 64.
    :param voc: Fully populated and pre-processed Voc object.
    :return: Input tensor ready for embedding, length of the sentences.
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in queries]
    sentc_lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)  # |v| x D dimensional matrix for embedding; each column is a padded sentence
    return padVar, sentc_lengths


def outputVar(replies: List[str], voc: Voc) -> Tuple[LongTensor, BoolTensor, int]:
    """
    :param replies: A batch of reply sentences. Default batch_size in train.py is 64.
    :param voc: Fully populated and pre-processed Voc object.
    :return: Padded target sequence tensor, padding mask, and max target length
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in replies]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc: Voc, pair_batch: List[List[str]]) -> Tuple[LongTensor, Tensor, LongTensor, BoolTensor, int]:
    """
    :param voc: Fully populated and pre-processed Voc object.
    :param pair_batch: A batch of query-reply pairs. Default batch_size in train.py is 64.
    :return: Input batch for embedding, length of each sentence in the batch, target batch, binary masking matrix,
    the length of the longest sentence in the batch.
    """
    # Sort batch of pairs in reverse according to length of the query sentences
    print("pair_batch:", pair_batch)
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus_name, datafile)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

if __name__ == '__main__':
    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)
