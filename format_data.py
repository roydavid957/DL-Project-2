import os
import re
import codecs
import csv
from typing import List, Dict, AnyStr
import itertools
import pprint


def loadLines(fileName: str, fields: List[str]) -> Dict[str, Dict[str, str]]:
    """ Splits each line of the file into a dictionary of fields\n
    :param fileName: Path to file to be read (movie_lines.txt).
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


def loadConversations(fileName: str, lines: Dict[str, Dict[str, str]], fields: List[str]) -> List[dict]:
    """ Groups fields of lines from `loadLines` into conversations\n*
    :param fileName: Path to file to be read (movie_conversations.txt).
    :param lines: Preferably the return value from 'loadLines'.
    :param fields: Name of columns to be assigned to fileName's content.
    :return:
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
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
        print("", "-" * 20, sep="\n")
        print("Conversation example, e.g. first row from movie_conversations.txt:")
        print("-" * 20)
        pprint.pprint(conversations[0])
    return conversations


def extractSentencePairs(conversations: List[dict]) -> List[List[str]]:
    """
    Extracts pairs of sentences from conversations
    :param conversations: A list of the pre-processed rows from movie_conversations.txt.
    :return: Query and Reply sentence pairs in a list.
    """
    qa_pairs = []
    for conversation in conversations:
        for i in range(0, len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()  # Query
            targetLine = conversation["lines"][i + 1]["text"].strip()  # Reply
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    print("", "-" * 20, sep="\n")
    print("Example Query & Reply sentence pairs:")
    print("-" * 20)
    print(qa_pairs[0], qa_pairs[1])
    return qa_pairs


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

if __name__ == '__main__':

    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new file, "formatted_movie_lines.txt"
    print("\nWriting newly formatted file...")
    delimiter = '\t'  # Signals the end of query and the start of response in the pair
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)
