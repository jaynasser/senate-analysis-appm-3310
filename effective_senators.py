from senate_data import create_senators_df, create_votes_df
from cosine_similarity import split_votes_df_by_party, create_vote_matrix

import numpy as np


def compute_num_effective_senators(matrix):

	vote_record_freqs = np.unique(matrix, return_counts=True, axis=0)[1] # frequencies of unique columns of vote matrix

	vote_record_relative_freqs = vote_record_freqs / sum(vote_record_freqs) # normalization

	information = -sum(
		vote_record_relative_freqs * np.log(vote_record_relative_freqs) / np.log(3) # information will be in base three (trits)
	)

	return information + 1


if __name__ == '__main__':

	df_senators = create_senators_df()
	df_votes = create_votes_df()

	df_votes_dem, df_votes_rep = split_votes_df_by_party(df_votes, df_senators)

	vote_mat_dem, vote_mat_rep = create_vote_matrix(df_votes_dem), create_vote_matrix(df_votes_rep)

	print(compute_num_effective_senators(vote_mat_rep), compute_num_effective_senators(vote_mat_dem))