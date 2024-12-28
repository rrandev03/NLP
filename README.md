## Repository Overview
This repository consolidates several assignments from an introduction to NLP course at Duke University, taught by Patrick Wang. It contains the following projects:

1. Markov Text Generator: Implements a basic Markov text generator in the *finish_sentence* function, which takes an input sentence, n-gram length, source corpus, and a flag indicating whether the generator should be deterministic or stochastic as arguments. Stupid backoff (a=0.4) is applied in order to generate a random sentence based on the seed text. A test script provided by Prof. Wang is used to test the generator's ability to complete sentences.
2. Spelling Corrector: Implements a spelling corrector in the *correct* function based on the noisy-channel model (unigram + weighted-Levenshtein-distance error model).
3. POS (part-of-speech) Tagger: Implements a part-of-speech hidden markov model. A Viterbi implementation created by Prof. Wang is used to evaluate the model by inferring the POS tags for a few sentences in the Brown corpus from the nltk library.
4. Gradient Descent: Includes *probabilities_plot* and *loss_plot* functions for visualizing learned token probabilities and training loss, from a neural network implementation (provided by Prof. Wang) for a unigram model.
5. LDA: Uses the Latent Dirichlet Allocation model to implement the *lda_gen* function, which takes alpha, beta, document size, and the vocabulary as arguments and returns an output text. A test function provided by Prof. Wang is also used in this script to generate a corpus of texts using lda_gen, and then use gensim to infer the topic distribution of these texts (the analysis of how well this maps to the original distribution is discussed in the pdf).  

For any questions or comments, please feel free to reach out to me at rishika.randev@duke.edu!
