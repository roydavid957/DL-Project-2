import os
import re
import codecs
import csv
from typing import List, Dict, Union, Tuple
import itertools
import pprint


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
        lines_head = dict(itertools.islice(lines.items(), 2))
        print("", "-" * 20, sep="\n")
        print("Lines example:")
        print("-" * 20)
        pprint.pprint(lines_head)

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
        print("", "-" * 20, sep="\n")
        print("Conversation example, e.g. first row from movie_conversations.txt:")
        print("-" * 20)
        pprint.pprint(conversations[0])
    return conversations


def extract_and_split_sentence_pairs(conversations: List[dict]) -> Tuple[List[list], Dict[str, list]]:
    """
    Extracts pairs of sentences from conversations
    :param conversations: A list of the pre-processed rows from movie_conversations.txt,
     a.k.a. output of loadConversations.
    :return: Query and Reply sentence pairs in a list.
    """
    qa_pairs_all = []
    qa_pairs_split = {"train": [], "val": [], "test": []}
    utterances = 442564  # number was retrieved before running the function without split
    count = 0
    for conversation in conversations:
        train_qa_pairs = []
        val_qa_pairs = []
        test_qa_pairs = []
        qa_pairs = []
        for i in range(0, len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()  # Query
            targetLine = conversation["lines"][i + 1]["text"].strip()  # Reply

            # Filter wrong samples (if one of the lists is empty); Split the data
            if inputLine and targetLine:
                pair = [inputLine, targetLine]
                qa_pairs.append(pair)
                if count < int(utterances * 0.6):
                    train_qa_pairs.append(pair)
                elif int(utterances * 0.6) < count < int(utterances * 0.8):
                    train_qa_pairs.append(pair)
                else:
                    test_qa_pairs.append(pair)
                count += 2
        qa_pairs_all.append(qa_pairs)
        if train_qa_pairs:
            qa_pairs_split["train"].append(train_qa_pairs)
        elif val_qa_pairs:
            qa_pairs_split["val"].append(val_qa_pairs)
        else:
            qa_pairs_split["test"].append(test_qa_pairs)

    print("", "-" * 20, sep="\n")
    print("Example Query & Reply sentence pairs:")
    print("-" * 20)
    print(qa_pairs_all[0], qa_pairs_all[1])

    return qa_pairs_all, qa_pairs_split


def extract_and_split_conversation_lines(conversations: List[dict]) -> Tuple[list, Dict[str, list]]:
    """
    :param conversations: A list of the pre-processed rows from movie_conversations.txt,
     a.k.a. output of loadConversations.
    :return: The dataset split into train/val/test sets according to 60%/20%/20% conventional split
    """
    rows_all = []
    rows_split = {"train": [], "val": [], "test": []}
    utterances = 304713
    count = 0
    for conversation in conversations:
        train_lines = []
        val_lines = []
        test_lines = []
        rows = []
        for i in range(len(conversation["lines"])):
            line = conversation["lines"][i]["text"].strip()
            rows.append(line)
            if count < int(utterances*0.6):
                train_lines.append(line)
            elif int(utterances*0.6) < count < int(utterances*0.8):
                train_lines.append(line)
            else:
                test_lines.append(line)
            count += 1
        rows_all.append(rows)

        if train_lines:
            rows_split["train"].append(train_lines)
        elif val_lines:
            rows_split["val"].append(val_lines)
        else:
            rows_split["test"].append(test_lines)

    return rows_all, rows_split


def write_data(new_file: str, conversations, query_reply):
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
    "val": os.path.join(split_path, "formatted_movie_lines_valid.txt"),
    "test": os.path.join(split_path, "formatted_movie_lines_test.txt"),
    "qr": os.path.join(split_path, "formatted_movie_QR_lines.txt"),
    "qr_train": os.path.join(split_path, "formatted_movie_QR_lines_train.txt"),
    "qr_val": os.path.join(split_path, "formatted_movie_QR_lines_valid.txt"),
    "qr_test": os.path.join(split_path, "formatted_movie_QR_lines_test.txt")
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

    conversations_all, conversations_split = extract_and_split_conversation_lines(conversations)
    # Sanity check: Number of lines should be distributed in a 60/20/20 ratio
    total_lines_train = sum(
        [len(conversations_split["train"][idx]) for idx, conversation in enumerate(conversations_split["train"])])
    total_lines_val = sum(
        [len(conversations_split["val"][idx]) for idx, conversation in enumerate(conversations_split["val"])])
    total_lines_test = sum(
        [len(conversations_split["test"][idx]) for idx, conversation in enumerate(conversations_split["test"])])
    print("total_lines_train:", total_lines_train)
    print("total_lines_val:", total_lines_val)
    print("total_lines_test:", total_lines_test)

    write_data(datafiles["default"], conversations_all, query_reply=False)
    write_data(datafiles["train"], conversations_split["train"], query_reply=False)
    write_data(datafiles["val"], conversations_split["val"], query_reply=False)
    write_data(datafiles["test"], conversations_split["test"], query_reply=False)

    qa_pairs_all, qa_pairs_split = extract_and_split_sentence_pairs(conversations)
    # Sanity check 2
    total_pairs_train = sum(
        [len(qa_pairs_split["train"][idx]) for idx, conversation in enumerate(qa_pairs_split["train"])])
    total_pairs_val = sum(
        [len(qa_pairs_split["val"][idx]) for idx, conversation in enumerate(qa_pairs_split["val"])])
    total_pairs_test = sum(
        [len(qa_pairs_split["test"][idx]) for idx, conversation in enumerate(qa_pairs_split["test"])])
    write_data(datafiles["qr"], qa_pairs_all, query_reply=True)
    write_data(datafiles["qr_train"], qa_pairs_split["train"], query_reply=True)
    write_data(datafiles["qr_val"], qa_pairs_split["val"], query_reply=True)
    write_data(datafiles["qr_test"], qa_pairs_split["test"], query_reply=True)
