"""Template provided by Patrick Wang."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp

    # for the first part, input is V X T data from one hot encodings. summing each row gives you V X 1 vector. transpose to get 1 X V vector, which basically just represents the count of each token in V from the training data.
    # logp is V X 1 vector of individual log token probabilites (weights)
    # so you end up with a 1 X 1 vector which is just log sequence probability (because if token 0 appears 7 times, and its log(p0) value = 0.2, log(p0^7) = 7*log(p0) and that's what the first value in the matrix multiplication gives you. And then you sum that with the next value because log(p0^7 * p1^3) = log(p0^7) + log(p1^3), just as an example assuming the next token has a count of 3).


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 50  # SET THIS
    learning_rate = 0.1  # SET THIS
    loss_list = []

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        logp_pred = model(
            x
        )  # Using the model.forward() method, evaluate our unigram model (which assigns probabilities to each char in the vocab) on the set of observations in x (the training data characters).

        loss = loss_fn(
            logp_pred
        )  # Evaluate loss fxn at this log of probability for the whole sequence of tokens based on the individual token probabilities from the unigram model
        loss_list.append(loss[0].detach())
        loss.backward()  # calculate the derivative of the loss
        optimizer.step()  # step (change the weights / token probabilities) in the direction that will minimize loss
        optimizer.zero_grad()  # clear out the gradient of the loss

    # get optimal probs
    optimal_p, sequence_p = optimal_probabilities(encodings)
    optimal_loss = loss_fn(sequence_p)

    # display results
    probabilities_plot(model, vocabulary, tokens, optimal_p)
    loss_plot(loss_list, optimal_loss)


def optimal_probabilities(encodings):
    """Calculate the optimal probabilities of each token based on their counts in the training data."""
    optimal_ps = np.sum(encodings, axis=1)
    total_count = np.sum(optimal_ps)
    for idx in range(len(optimal_ps)):
        optimal_ps[idx] /= total_count
    logged_probs = np.log(optimal_ps)
    sequence_p = np.sum(
        logged_probs @ encodings
    )  # sum the logged probabilities of each token in the sequence to get the logged prob of the whole sequence (the training data)
    return optimal_ps, sequence_p


def probabilities_plot(model, vocabulary, tokens, optimal):
    """Plot the final token probabilities after 50 iterations and the optimal probabilities."""
    final_probs = (torch.nn.Softmax(0)(model.s)).detach().numpy()
    tokens = vocabulary[:-2] + ["None", "Other"]
    plt.plot(tokens, final_probs, label="Probabilities After 50 Iterations")
    plt.plot(tokens, optimal, label="Optimal Probabilities")
    plt.xlabel("Token")
    plt.ylabel("Probability of Token")
    plt.title("Probability of Tokens ")
    plt.legend(loc="upper left")
    plt.show()


def loss_plot(loss_list, optimal_loss):
    """Plot the loss over 50 iterations along with the minimal loss that can be achieved."""
    plt.plot(loss_list, label="Loss From NN")
    plt.axhline(
        y=optimal_loss, color="orange", linestyle="-", label="Minimum possible loss"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()
