import numpy as np
from copy import deepcopy

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """
    Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # Initialize each label with its emission score at 0 and start score
    vit_y = [[[x], emission_scores[0][x] + start_scores[x]] for x in range(L)]

    for n in range(1, N):
        vit_prime = deepcopy(vit_y)
        for y in range(L):
            # Create pairs representing previous paths in the form of, [previous path index, previous path score]
            scores = [vit_prime[x][1] + trans_scores[x][y] + emission_scores[n][y] for x in range(L)]

            # Find the best scoring one
            best_score = max(scores)
            best_score_index = scores.index(best_score)

            # Update the path and score with it
            vit_y[y][0] = vit_prime[best_score_index][0] + [y]
            vit_y[y][1] = best_score

    # Add the end scores
    for l in range(L):
        vit_y[l][1] += end_scores[l]

    # Find the most likely sentence
    max_y = sorted(vit_y, key=lambda x: x[1], reverse=True)[0]

    # Return the score and path of the most likely sentence
    return (max_y[1], max_y[0])

