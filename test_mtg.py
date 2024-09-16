"""Markov Text Generator.

Patrick Wang, 2024

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""

import csv
import nltk


# from Markov.mtg import finish_sentence
from mtg import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = tuple(
        nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    )

    # with open("test_examples.csv") as csvfile:
    with open("test_examples.csv") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=",")
        for row in csvreader:
            words = finish_sentence(
                row["input"].split(" "),
                int(row["n"]),
                corpus,
                randomize=False,
            )
            print(f"input: {row['input']} (n={row['n']})")
            print(f"output: {' '.join(words)}")
            assert words == row["output"].split(" ")

    print("\nAdditional tests: \n")
    inputs_list = [
        ["they were going", 3, True],
        ["he should try to", 2, False],
        ["why would you take", 4, True],
        ["to be or not to be", 6, True],
        ["I am", 6, False],
    ]
    for input in inputs_list:
        words = finish_sentence(sentence=input[0].split(' '), n=input[1], corpus=corpus, randomize=input[2])
        print(f"input: {input}")
        print(f"output: {' '.join(words)}")
        print()
        

if __name__ == "__main__":
    test_generator()
