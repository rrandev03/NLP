import numpy as np
import csv


def sub_channel_model(x, y, subs, unigrams):
    """Returns the probability of a particular substitution sub(x,y) occurring based on
    the data in the substitutions.csv and unigrams.csv files."""
    if (x, y) in subs:
        count = subs[(x, y)]
    else:
        count = 0  # if we don't have a count of the substitution in the subs dictionary, set the count to 0
    opportunities = unigrams[x]
    return int(count) / int(opportunities)


def deletion_channel_model(x, y, deletions, bigrams):
    """Returns the probability of a particular deletion del(x,y) occurring based on
    the data in the deletions.csv and bigrams.csv files."""
    if (x, y) in deletions and x != "#":
        count = deletions[(x, y)]
        opportunities = bigrams[
            x + y
        ]  # add the strings together to get the original bigram.
        return int(count) / int(opportunities)
    else:  # either this deletion does not exist in our dictionary, or the prefix of the deletion is the start of word for which we don't have bigram counts, so return 0 in either case.
        return 0


def insertion_channel_model(x, y, additions, unigrams):
    """Returns the probability of a particular insertion ins(x,y) occurring based on
    the data in the additions.csv and unigrams.csv files."""
    if (x, y) in additions and x != "#":
        count = additions[(x, y)]
        opportunities = unigrams[x]
        return int(count) / int(opportunities)
    else:  # either this insertion does not exist in our dictionary, or the prefix of the insertion is the start of word for which we don't have unigram counts, so return 0 in either case.
        return 0


def read_csv(file_list):
    """Given a list of csv files, create a dictionary for each file, where the keys are the edits /
    letter patterns within the file and the values are the counts of their occurrences.
    Append all dictionaries to a list and return it."""
    dict_list = []
    for file in file_list:
        dict = {}
        with open(file) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            for row in csvreader:
                if len(row) == 2:
                    dict[row[0]] = row[1]
                else:  # for substitutions, additions, and deletions csv, there are 3 entries per row so use the first 2 as the key and the last (count) as the value.
                    dict[tuple(row[:-1])] = row[-1]
        dict_list.append(dict)
    return dict_list


def unigram_counts(file):
    """Based on data from the counts.txt file, this creates a dictionary where keys are the
    words in the counts file and the corresponding values are their counts."""
    counts_dict = {}
    with open(file) as text_file:
        for row in text_file:
            word, count = row.split("\t")
            count_int = int(count.strip("\n"))
            counts_dict[word] = count_int
    return counts_dict


def weighted_levenshtein_distance(w, x, deletions, bigrams, additions, unigrams, subs):
    """Returns weighted Levenshtein distance between source=w and target=x
    where weights correspond to the probability of the edit based on the provided csv files.
    """
    rows = len(w) + 1
    columns = len(x) + 1
    matrix = np.empty((rows, columns))
    matrix[0, 0] = 0
    w = "#" + w  # add start-of-word markers to original, x, and candidate, w
    x = "#" + x

    for i in range(1, rows):  # initialize first column with probabilities
        deletion_prob = deletion_channel_model(w[i - 1], w[i], deletions, bigrams)
        matrix[i, 0] = matrix[i - 1, 0] + deletion_prob
    for j in range(1, columns):  # initialize first row with probabilities
        insertion_prob = insertion_channel_model(x[j - 1], x[j], additions, unigrams)
        matrix[0, j] = matrix[0, j - 1] + insertion_prob

    for i in range(1, rows):  # populate rest of matrix
        for j in range(1, columns):
            sub_prob = sub_channel_model(w[i], x[j], subs, unigrams)
            deletion_prob = deletion_channel_model(w[i - 1], w[i], deletions, bigrams)
            insertion_prob = insertion_channel_model(
                x[j - 1], x[j], additions, unigrams
            )
            sub = (
                matrix[i - 1, j - 1] + sub_prob
            )  # get cumulative probabilities of substitution, deletion, and insertion
            delete = matrix[i - 1, j] + deletion_prob
            insert = matrix[i, j - 1] + insertion_prob
            best_prob = max(sub, delete, insert)
            matrix[i, j] = (
                best_prob  # set this cell in the matrix to the highest cumulative probability
            )
    return matrix[rows - 1, columns - 1]


def levenshtein_distance(w, x):
    """Returns minimum edit distance between source=w and target=x."""
    rows = len(w) + 1
    columns = len(x) + 1
    matrix = np.empty((rows, columns))
    matrix[0, 0] = 0

    for i in range(1, rows):
        matrix[i, 0] = matrix[i - 1, 0] + 1
    for j in range(1, columns):
        matrix[0, j] = matrix[0, j - 1] + 1

    for i in range(1, rows):
        for j in range(1, columns):
            sub = matrix[i - 1, j - 1]
            delete = matrix[i - 1, j]
            insert = matrix[i, j - 1]
            mde_prior = min(sub, delete, insert)
            if (
                w[i - 1] == x[j - 1] and mde_prior == sub
            ):  # This is done because the substitution cost for an identical letter is 0, so we can just use the previous substitution distance if it is the lowest out of all the surrounding distances.
                matrix[i, j] = mde_prior
            else:
                matrix[i, j] = mde_prior + 1
    return matrix[rows - 1, columns - 1]


def correct(original: str) -> str:
    """Returns a corrected word based on a misspelled original word. The corrected word will be within 1 edit of the
    original and will be the most likely correction based on a unigram model and weighted Levenshtein edit distance.
    If there is no real word within 1 edit of the original, then the function does not provide a correction.
    """
    additions, bigrams, deletions, subs, unigrams = read_csv(
        [
            "additions.csv",
            "bigrams.csv",
            "deletions.csv",
            "substitutions.csv",
            "unigrams.csv",
        ]
    )
    len_original = len(original)
    candidates_list = []
    counts_dict = unigram_counts("counts.txt")
    sum_words = sum(counts_dict.values())
    for word in counts_dict.keys():
        len_word = len(word)
        if (
            len_original == len_word + 1
            or len_original == len_word - 1
            or len_original == len_word
        ):  # since we're only considering words within 1 edit of the original, we can screen based on the length of words first
            min_edit_distance = levenshtein_distance(word, original)
            if min_edit_distance == 1.0:
                candidates_list.append(
                    word
                )  # add any word within 1 edit of the original to the candidate list
    max_prob = -1
    best_word = "No candidates within one edit of this word"
    for candidate in candidates_list:
        prior = counts_dict[candidate] / sum_words
        channel = weighted_levenshtein_distance(
            candidate, original, deletions, bigrams, additions, unigrams, subs
        )
        prob = prior * channel
        if prob > max_prob:
            max_prob = prob
            best_word = candidate
    return best_word


if __name__ == "__main__":
    print(
        f"Original: The old house are very spacious.\nCorrection: The old {correct('house')} are very spacious."
    )
    # print(f"Original: the\nCorrection: {correct('the')}")
    # print(f"Original: aubergii\nCorrection: {correct('aubergii')}")
    # print(f"Original: inser\nCorrection: {correct('inser')}")
    # print(f"Original: workingf\nCorrection: {correct('workingf')}")
    # print(f"Original: tiied\nCorrection: {correct('tiied')}")
