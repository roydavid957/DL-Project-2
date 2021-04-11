import os
import re
import pickle
import csv
from typing import List, Dict, Union, Tuple
import itertools
import pprint

from build_vocabulary import loadPrepareData, trimRareWords, MIN_COUNT

def loadLines(fileName: str, fields: List[str]) -> Dict[str, Dict[str, str]]:
    """ Splits each line of the file into a dictionary of fields
    :param fileName: Path to the file to be read (movie_lines.txt).
    :param fields: Name of columns to be assigned to fileName's content.
    :return: A dict with keys in the format of ['L1045', 'L1044', ...,]
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {field: values[i] for i, field in enumerate(fields)}
            lines[lineObj['lineID']] = lineObj

        # Sanity check
        # lines_head = dict(itertools.islice(lines.items(), 2))
        # print("", "-" * 20, sep="\n")
        # print("Lines example:")
        # print("-" * 20)
        # pprint.pprint(lines_head)

    return lines


def loadConversations(fileName: str, loaded_lines: Dict[str, Dict[str, str]], fields: List[str]) -> List[dict]:
    """ Groups fields of lines from `loadLines` into conversations\n*
    :param fileName: Path to file to be read (movie_conversations.txt).
    :param loaded_lines: Preferably the return value from 'loadLines'.
    :param fields: Name of columns to be assigned to fileName's content.
    :return: A list of the conversations as dictionaries, with the 'lines' key within each dictionary
     containing a list of dictionaries, where the actual lineID and its text can be found.
    """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {field: values[i] for i, field in enumerate(fields)}

            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])

            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(loaded_lines[lineId])
            conversations.append(convObj)
        # print("", "-" * 20, sep="\n")
        # print("Conversation example, e.g. first row from movie_conversations.txt:")
        # print("-" * 20)
        # pprint.pprint(conversations[0])
    return conversations


def extract_and_split_sentence_pairs(conversations: List[dict]) -> List[list]:
    """
    Extracts pairs of sentences from conversations
    :param conversations: A list of the pre-processed rows from movie_conversations.txt,
     a.k.a. output of loadConversations.
    :return: Query and Reply sentence pairs in a list.
    """
    qa_pairs_all = []
    for conversation in conversations:
        qa_pairs = []
        for i in range(0, len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()  # Query
            targetLine = conversation["lines"][i + 1]["text"].strip()  # Reply
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
        qa_pairs_all.append(qa_pairs)

    # print("", "-" * 20, sep="\n")
    # print("Example Query & Reply sentence pairs:")
    # print("-" * 20)
    # print(qa_pairs_all[0], qa_pairs_all[1])

    return qa_pairs_all


def split(pairs):
    """
    :param pairs: Entire dataset
    """
    pair_num = len(pairs)
    data = {"train": [], "test": []}
    for idx, pair in enumerate(pairs):
        if idx < int(pair_num*0.8):
            data["train"].append(pair)
        else:
            data["test"].append(pair)
    return data


def write_data(new_file: str, conversations, query_reply=False):
    """ Write new file from extracted conversations,
     e.g. "formatted_movie_convQR_lines.txt", "formatted_movie_conv_lines.txt """

    print("\nWriting newly formatted file...")
    delimiter = '\t'  # Signals the end of query and the start of response in the pair
    with open(new_file, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        if query_reply:
            for conversation in conversations:
                for qa_pair in conversation:
                    writer.writerow(qa_pair)
        else:
            for conversation in conversations:
                writer.writerow(conversation)


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
split_path = os.path.join("data", "train_test_split")

# Define path to new files
datafiles = {
    "default": os.path.join(split_path, "formatted_movie_lines.txt"),
    "train": os.path.join(split_path, "formatted_movie_lines_train.txt"),
    "test": os.path.join(split_path, "formatted_movie_lines_test.txt"),

    "qr": os.path.join(split_path, "formatted_movie_QR_lines.txt"),
    "qr_train": os.path.join(split_path, "formatted_movie_QR_lines_train.txt"),
    "qr_test": os.path.join(split_path, "formatted_movie_QR_lines_test.txt"),
}


if __name__ == '__main__':

    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    qr_pairs = extract_and_split_sentence_pairs(conversations)
    write_data(datafiles["qr"], qr_pairs, query_reply=True)
    voc, pairs = loadPrepareData(datafiles["qr"])
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    split_pairs = split(pairs)
    write_data(datafiles["qr_train"], split_pairs["train"], query_reply=False)
    write_data(datafiles["qr_test"], split_pairs["test"], query_reply=False)
    with open(f"{os.path.join(split_path, 'voc.pickle')}", "wb") as f:
        pickle.dump(voc, f)
