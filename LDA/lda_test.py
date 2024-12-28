"""Latent Dirichlet Allocation:

Template provided by Patrick Wang, 2021.
"""

from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(
    vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int
) -> List[str]:
    dirichlet = np.random.dirichlet(alpha)
    doc_length = np.random.poisson(xi)
    words = []
    for _ in range(doc_length):
        topic_vector = np.random.multinomial(1, dirichlet)
        t_index = list(topic_vector).index(1)
        word_vector = np.random.multinomial(1, beta[t_index])
        word_index = list(word_vector).index(1)
        word = vocabulary[word_index]
        words.append(word)
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass",
        "pike",
        "deep",
        "tuba",
        "horn",
        "catapult",
    ]
    beta = np.array(
        [
            [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
            [0.3, 0.0, 0.2, 0.3, 0.2, 0.0],
        ]
    )
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [lda_gen(vocabulary, alpha, beta, xi) for _ in range(100)]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()
