import numpy as np
import csv

"""This is a spelling corrector which uses Levenshtein distance to get the candidates within 1 edit
of the misspelling and also keeps track of the edit. It then calculates the probability of that edit
using the channel_model function (instead of doing the full weighted Levenshtein distance calculation). 
spelling_corrector_alternative.py is an alternative corrector which uses weighted Levenshtein."""


def channel_model(before, after, additions, bigrams, deletions, subs, unigrams):
    """Returns the probability of a specific edit based on the provided csv files."""
    if len(before) == len(after) and (before, after) in subs:
        # it's a substitution. find count of times before got subbed as after
        count = subs[(before, after)]
        opportunities = unigrams[before]
    elif (
        len(before) == len(after) + 1 and before in deletions
    ):  # deletion that exists in our dictionary
        if (
            before[0] == "#"
        ):  # this deletion occurred at start of word and we don't have opportunity counts for that, so just return 0
            return 0
        else:
            count = deletions[before]
            opportunities = bigrams["".join(before)]
    elif (
        len(before) == len(after) - 1 and after in additions
    ):  # insertion that exists in our dictionary
        if (
            before == "#"
        ):  # this insertion occurred at start of word and we don't have opportunity counts for that, so just return 0
            return 0
        else:
            count = additions[after]
            opportunities = unigrams[before]
    else:
        return 0
    return int(count) / int(opportunities)


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


def find_edit(matrix, w, x):
    """Using a given edit matrix for source=w and target=x which have an edit distance of 1, this
    function returns the details of the specific edit as a tuple, where the first element
    is the character(s) before the edit occurred and the second element is the character(s) after the edit occurred.
    """
    i = len(w)
    j = len(x)
    surrounding_insert = matrix[i, j - 1]
    surrounding_delete = matrix[i - 1, j]
    surrounding_sub = matrix[i - 1, j - 1]
    while surrounding_insert != 0 and surrounding_delete != 0 and surrounding_sub != 0:
        i -= 1
        j -= 1
        surrounding_insert = matrix[i, j - 1]
        surrounding_delete = matrix[i - 1, j]
        surrounding_sub = matrix[i - 1, j - 1]
    if surrounding_insert == 0:
        before = w[i - 1]
        after = (w[i - 1], x[j - 1])
    elif surrounding_delete == 0:
        before = (w[i - 2], w[i - 1])
        after = w[i - 2]
    else:
        before = w[i - 1]
        after = x[j - 1]
    return before, after


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
            if w[i - 1] == x[j - 1] and mde_prior == sub:
                matrix[i, j] = (
                    mde_prior  # This is done because the substitution cost for an identical letter is 0, so we can just use the previous substitution distance if it is the lowest out of all the surrounding distances.
                )
            else:
                matrix[i, j] = mde_prior + 1
    return matrix, matrix[rows - 1, columns - 1]


def correct(original: str) -> str:
    """Returns a corrected word based on a misspelled original word. The corrected word will be within 1 edit of the
    original and will be the most likely correction based on a unigram model and the probability of the specific edit required.
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
    candidates_dict = {}
    counts_dict = unigram_counts("counts.txt")
    sum_words = sum(counts_dict.values())
    for word in counts_dict.keys():
        len_word = len(word)
        if (
            len_original == len_word + 1
            or len_original == len_word - 1
            or len_original == len_word
        ):  # since we're only considering words within 1 edit of the original, we can screen based on the length of words first
            matrix, min_edit_distance = levenshtein_distance(word, original)
            if min_edit_distance == 1.0:
                before, after = find_edit(matrix, word, original)
                candidates_dict[word] = (
                    before,
                    after,
                )  # add any word within 1 edit of the original to the candidate dict and save its edit as the value
    max_prob = -1
    best_word = "No candidates within one edit of this word"
    for candidate in candidates_dict:
        prior = counts_dict[candidate] / sum_words
        before = candidates_dict[candidate][0]
        after = candidates_dict[candidate][1]
        channel = channel_model(
            before, after, additions, bigrams, deletions, subs, unigrams
        )
        prob = prior * channel
        if prob > max_prob:
            max_prob = prob
            best_word = candidate
    return best_word


if __name__ == "__main__":
    print(
        f"Original: The small house ran across the room.\nCorrection: The small {correct('house')} ran across the room."
    )
    print(f"Original: the\nCorrection: {correct('the')}")
    print(f"Original: aubergii\nCorrection: {correct('aubergii')}")
    print(f"Original: inser\nCorrection: {correct('inser')}")
    print(f"Original: workingf\nCorrection: {correct('workingf')}")
    print(f"Original: tiied\nCorrection: {correct('tiied')}")
