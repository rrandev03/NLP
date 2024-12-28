import nltk
import numpy as np
from viterbi import viterbi

nltk.download("brown")
nltk.download("universal_tagset")

training = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
testing = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]


def create_mappings(training):
    """This function maps each unique word in the training data to an integer and stores
    this mapping in a dictionary obs_mappings, with one added observation for unknown words (UNK) that
    we might see in the testing data. It also maps each unique POS to an integer and stores this
    in a separate dictionary state_mappings."""
    state_mappings = {}
    obs_mappings = {}
    count_unique_states = 0
    count_unique_obs = 0
    for sentence in training:
        for tup in sentence:
            word = tup[0]
            pos = tup[1]
            if pos not in state_mappings:
                state_mappings[pos] = count_unique_states
                count_unique_states += 1
            if word not in obs_mappings:
                obs_mappings[word] = count_unique_obs
                count_unique_obs += 1
    obs_mappings["UNK"] = count_unique_obs
    return state_mappings, obs_mappings


def pos_from_int(pos_int, state_mappings):
    """This function takes a list of integers and returns
    a list of the corresponding POS for those integers by reversing
    the state_mappings dictionary. This is used during the validation of
    the predicted POS sequence from the Viterbi algorithm."""
    pos_list = []
    reverse_map = {value: key for key, value in state_mappings.items()}
    for int in pos_int:
        pos_list.append(reverse_map[int])
    return pos_list


def fill_matrices(training, initial_matrix, emissions_matrix, transition_matrix):
    """This function takes in initialized matrices A & B, of appropriate
    shape to represent transition probabilities between POS and emission
    probabilities between words/observations and POS/states, respectively. It also
    takes in an initialized pi vector which will represent the initial probabilities of
    the POS/states. It then populates all of these matrices appropriately with counts
    and finally converts these counts into probabilities."""
    for sentence in training:
        start_pos = sentence[0][1]
        start_idx = state_mappings[start_pos]
        initial_matrix[start_idx] += 1
        start_word = sentence[0][0]
        word_idx = obs_mappings[start_word]
        emissions_matrix[start_idx][word_idx] += 1
        for tup in sentence[1:]:
            word = tup[0]
            pos = tup[1]
            pos_idx = state_mappings[pos]
            word_idx = obs_mappings[word]
            emissions_matrix[pos_idx][word_idx] += 1
            transition_matrix[start_idx][pos_idx] += 1
            start_pos = pos
            start_idx = pos_idx
    count_pi = sum(initial_matrix)
    for i in range(len(initial_matrix)):
        initial_matrix[i] /= count_pi
    for row in transition_matrix:
        count_initial_pos = sum(row)
        for i in range(len(row)):
            row[i] /= count_initial_pos
    for row in emissions_matrix:
        count_pos = sum(row)
        for i in range(len(row)):
            row[i] /= count_pos


if __name__ == "__main__":
    state_mappings, obs_mappings = create_mappings(training)
    transition_matrix = np.ones(
        shape=(len(state_mappings), len(state_mappings))
    )  # Initialized with 1s for add-1 smoothing
    emissions_matrix = np.ones(
        shape=(len(state_mappings), len(obs_mappings))
    )  # Initialized with 1s for add-1 smoothing
    initial_matrix = np.zeros(
        shape=(len(state_mappings),)
    )  # Not initialized with 1s because none of the initial POS counts are 0, so add-1 smoothing is not required
    fill_matrices(training, initial_matrix, emissions_matrix, transition_matrix)
    for sentence in testing:
        print(sentence)
        list_test_obs = []
        list_actual_pos = []
        for tuple in sentence:
            actual_pos = tuple[1]
            list_actual_pos.append(actual_pos)
            word = tuple[0]
            word_map = (
                obs_mappings[word] if word in obs_mappings else obs_mappings["UNK"]
            )
            list_test_obs.append(word_map)
        pos_seq, prob = viterbi(
            list_test_obs, initial_matrix, transition_matrix, emissions_matrix
        )
        list_predicted_pos = pos_from_int(pos_seq, state_mappings)
        print("Actual taggings: ")
        print(list_actual_pos)
        print("Predicted taggings:")
        print(list_predicted_pos, end="\n\n")

      
