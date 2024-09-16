import numpy as np
import random

def build_distribution(context, corpus, vocab, backoffs):
    """Builds initial probability distribution based on the provided
    context / provided number of backoffs using an alpha of 0.4. If any of the words' 
    probabilities are 0,  backoff is triggered to make them non-zero. To do this, 
    the lower n-gram probability distribution is built and scaled appropriately by 
    alpha, and then the stupid_backoff function is used to replace the initially 0 
    probabilities with the lower n-gram probability. If any probabilities are still 
    0, we repeat, continuing all the way to the unigram where we have an 
    empty context.""" 
    alpha = 0.4
    probabilities = get_probabilities(context, corpus, vocab, alpha, backoffs)
    sub_context = context[1:]
    while 0 in probabilities.values():
        backoffs += 1
        sub_probabilities = get_probabilities(sub_context, corpus, vocab, alpha, backoffs)
        stupid_backoff(probabilities, sub_probabilities)
        sub_context = sub_context[1:]
    return probabilities

def get_probabilities(context, corpus, vocab, alpha, backoffs):
    """Initializes a dictionary with all the vocabulary words set to counts of 0. 
    Searches for a context in the corpus. When it's found, it takes the word following 
    the context and increments its count in the dictionary. Then divides the counts
    by the total number of times the context was found, assuming the context was found
    at least once, and scales by alpha appropriately if this is a backoff scenario,
    to get the probability of each word occurring after this context. This is 
    probably not a computationally efficient method because we are iterating through
    the corpus every time we need to generate a new word, and sometimes multiple 
    times for a single word if we have to backoff and use a smaller context."""
    n_1 = len(context)
    context_counter = 0
    probabilities = {word:0 for word in vocab}
    for idx in range(len(corpus) - n_1):
        if corpus[idx : idx + n_1] == context:
            word = corpus[idx + n_1]
            probabilities[word] += 1
            context_counter += 1
    if context_counter != 0:
        for word in probabilities:
            probabilities[word] = (alpha**backoffs) * probabilities[word] / context_counter
    return probabilities

def stupid_backoff(probabilities, sub_probabilities): 
    """Replaces any zero probabilities in the probabilities dictionary
    with the corresponding word's probability from sub_probabilities."""
    for word, probability in probabilities.items():
        if probability == 0:
            probabilities[word] = sub_probabilities[word]

def finish_sentence(sentence, n, corpus, randomize=False): 
    vocab = set(corpus)
    finished_sentence = list(sentence)
    next_word = ""
    next_word_index = len(sentence) 
    while next_word not in [".", "!", "?"] and next_word_index != 10:
        if next_word_index < n - 1: #need to do immediate backoff because we don't have enough words in the sentence
            context = finished_sentence
            backoffs = n - 1 - next_word_index #ex. for an n=5-gram you need 4 word context. if sentence length = 3, the best you can do is a 4-gram so you have to backoff by 1.
        else:
            context = finished_sentence[
            next_word_index - n + 1 : next_word_index]  # find context based on n and existing sentence list
            backoffs = 0
        prob_dict = build_distribution(tuple(context), corpus, vocab, backoffs)
        if randomize == True:
            words = list(prob_dict.keys())
            probs = list(prob_dict.values())
            next_word = ''.join(random.choices(words, probs, k=1))
            # take in the prob dict, and use the values as weights for random.choices() to get the next_word
        else:
            potentials = []
            max_prob = max(prob_dict.values())
            for word, probability in prob_dict.items():
                if probability == max_prob:
                    potentials.append(word)
            next_word = (sorted(potentials))[0]       
            # take max value from the prob dict, and set its key as next_word
        finished_sentence.append(next_word)
        next_word_index += 1
    return finished_sentence