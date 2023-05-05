from cosine_similarity import create_vote_matrix, split_votes_df_by_party
from senate_data import create_senators_df, create_votes_df

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def compute_num_effective_senators(matrix: np.ndarray) -> float:

	vote_record_freqs = np.unique(matrix, return_counts=True, axis=1)[1] # frequencies of unique rows of vote matrix

	vote_record_relative_freqs = vote_record_freqs / sum(vote_record_freqs) # normalization

	information = -sum(
		vote_record_relative_freqs * np.log(vote_record_relative_freqs) / np.log(3) # information will be in base three (trits)
	)

	return information


def singular_value_decomposition(matrix: np.ndarray):

	singular_values, singular_vectors = la.svd(matrix)[1: 3]
	singular_vectors = singular_vectors.T
	
	if singular_vectors[0, 0] < 0:
		singular_vectors *= -1
	
	singular_value_freqs = singular_values / sum(singular_values)

	return singular_values, singular_value_freqs, singular_vectors


def approx_margin(vote_matrix: np.ndarray, singular_vectors: np.ndarray, n_terms: int = 10, round: bool = False) -> np.ndarray:

	singular_vectors_truncated = singular_vectors.T[:, :n_terms]
	coords = (vote_matrix @ singular_vectors.T)[:, :n_terms]

	n_rows, n_cols = coords.shape
	margins = np.empty((n_rows, ))
	for i in range(n_rows):
		margin = 0
		for j in range(n_cols):
			margin += sum(singular_vectors_truncated[:, j] * coords[i, j])
		margins[i] = margin
	if round:
		return np.round(margins)
	return margins


def compute_margin_error(vote_matrix: np.ndarray, singular_vectors: np.ndarray, n: int) -> int:
	
	actual_margins = np.sum(vote_matrix, axis=1)
	margin_approx = approx_margin(vote_matrix, singular_vectors, n_terms=n, round=True)

	return np.sum(np.abs(actual_margins - margin_approx))


def plot_margin_errors(vote_matrix: np.ndarray, singular_vectors: np.ndarray, party: Literal['Democrat', 'Republican'], saveout: bool = False) -> None:
	
	margin_errors = []
	senators_range = range(1, vote_matrix.shape[1])
	for n in senators_range:
		margin_errors.append(compute_margin_error(vote_matrix, singular_vectors.T, n))

	plt.plot(list(senators_range), margin_errors)
	plt.title(f'Errors for {party} Vote Margin Approximations')
	plt.xlabel('Dimension of Approximation')
	plt.ylabel('Margin Error')
	
	if saveout:
		plt.savefig(f'./{party}_margin_errors.png')
	plt.show()