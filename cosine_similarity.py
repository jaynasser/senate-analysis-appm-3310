from senate_data import create_senators_df, create_votes_df

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_votes_df_by_party(df_votes: pd.DataFrame, df_senators: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	
	def calculate_most_popular_nonzero_votes(votes: pd.Series) -> np.ndarray:

		counts = votes.value_counts()

		if len(counts) == 1:
			if counts.index[0] != 0:
				return np.array(counts.index[0])
			else:
				raise ValueError('Unable to compute most popular nonzero vote due to unanimous abstinence.')
		else:
			if counts.index[0] != 0:
				if counts.iloc[0] != counts.iloc[1]:
					return np.array(counts.index[0])
				else:
					return np.array((counts.index[0], counts.index[1]))
			else:
				return np.array(counts.index[1])


	icpsr_to_party = dict(zip(df_senators['icpsr'], df_senators['party_code']))
	
	df_votes_party = df_votes.copy()
	df_votes_party['party_code'] = df_votes.icpsr.map(icpsr_to_party)
	df_votes_party = df_votes_party.merge(
		df_votes_party.groupby(['party_code', 'rollnumber'])['cast_code'].apply(calculate_most_popular_nonzero_votes).rename(index='majority_vote'),
		how='left',
		left_on=['party_code', 'rollnumber'],
		right_index=True
	)

	df_votes_party['vote_relative_to_majority'] = df_votes_party.apply(lambda x: np.max(x.cast_code * x.majority_vote), axis=1)

	df_votes_party = df_votes_party[['rollnumber', 'icpsr', 'vote_relative_to_majority', 'cast_code', 'party_code']]

	df_votes_dem = df_votes_party[df_votes_party['party_code'] == 100]
	df_votes_rep = df_votes_party[df_votes_party['party_code'] == 200]
	
	return df_votes_dem, df_votes_rep


def create_vote_matrix(df_votes: pd.DataFrame, relative_votes: bool =True) -> np.ndarray:

	senator_vote_array = df_votes.to_numpy(dtype='int')

	num_rollcalls = len(df_votes['rollnumber'].unique()) # number of votes that occurred
	
	if relative_votes:
		vote_column_index = 2
	else:
		vote_column_index = 3
	
	vote_matrix = np.reshape(senator_vote_array[: , vote_column_index], (num_rollcalls, -1))

	return vote_matrix


def create_cosine_similarity_matrix(df_votes: pd.DataFrame, dissent_index: bool = False) -> np.ndarray:

	vote_matrix = create_vote_matrix(df_votes)

	d = vote_matrix.T @ vote_matrix
	norm = (vote_matrix * vote_matrix).sum(0, keepdims=True) ** .5
	cos_mat = d / norm / norm.T

	if dissent_index:
		return 1 - cos_mat
	else:
		return cos_mat


def cos_sim_mat_sum(matrix: np.ndarray, include_self: bool = True) -> np.ndarray:
	
	result = np.sum(matrix, axis=1) # both axes have the same sums since cos sim matrix is symmetric
	
	if include_self:
		return result
	else:
		return result - matrix[0, 0]


def fetch_relevant_names(df_senators: np.ndarray, df_votes: np.ndarray) -> np.ndarray:

	icpsr_to_name = dict(zip(df_senators['icpsr'], df_senators['bioname']))
	return np.vectorize(icpsr_to_name.get)(list(df_votes.groupby('icpsr').groups))


def compute_individual_similarities(matrix: np.ndarray, names: np.ndarray = None) -> np.ndarray:

	individual_similarities = cos_sim_mat_sum(matrix, include_self=False) / (len(matrix) - 1)
	
	if names is None:
		return sorted(individual_similarities)
	else:
		return np.array(
			sorted(
				list(
					zip(
						names,
						individual_similarities
					)
				),
				key=lambda x: x[1]
			)
		)


def plot_cosine_similarity_matrix(
	df_senators: pd.DataFrame,
	df_votes: pd.DataFrame,
	matrix: np.ndarray,
	saveout: bool = False,
	saveout_fp: str = './senate_cos_sim.png',
	figsize: Tuple[float, float] = (30, 30),
	cmap: str = 'hot',
	colorbar_min: float = 0.3
) -> None:

	plot_labels = fetch_relevant_names(df_senators, df_votes)

	fig = plt.figure(figsize=figsize)
	ax = plt.gca()

	im = ax.matshow(matrix, cmap=cmap)
	fig.colorbar(im, fraction=0.046, pad=0.04)
	im.set_clim(colorbar_min, 1)

	ax.set_xticks(np.arange(len(plot_labels)))
	ax.set_xticklabels(plot_labels, rotation=90)
	ax.set_yticks(np.arange(len(plot_labels)))
	ax.set_yticklabels(plot_labels)

	if saveout:
		plt.savefig(saveout_fp)

	plt.show()


def cos_sim(
	df_senators: pd.DataFrame = None,
	df_votes: pd.DataFrame = None,
	dissent_index: bool = False,
	saveout_plot: bool = False,
	saveout_fp: Tuple[str, str] = ('./senate_cos_sim_dem.png', './senate_cos_sim_rep.png')
) -> None:

	df_senators, df_votes_dem, df_votes_rep, cos_mat_dem, cos_mat_rep = create_all_matrices(
		df_senators=df_senators,
		df_votes=df_votes
	)

	cos_sim_min = np.min(
		[np.min(cos_mat_dem), np.min(cos_mat_rep)]
	)
	
	plot_cosine_similarity_matrix(
		df_senators,
		df_votes_dem,
		cos_mat_dem,
		saveout=saveout_plot,
		saveout_fp=saveout_fp[0],
		colorbar_min=cos_sim_min
	)
	plot_cosine_similarity_matrix(
		df_senators,
		df_votes_rep,
		cos_mat_rep,
		saveout=saveout_plot,
		saveout_fp=saveout_fp[1],
		colorbar_min=cos_sim_min
	)


def create_all_matrices(
	df_senators: pd.DataFrame = None,
	df_votes: pd.DataFrame = None,
	dissent_index: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	
	if df_senators is None:
		df_senators = create_senators_df()
	
	if df_votes is None:
		df_votes = create_votes_df()
	
	df_votes_dem, df_votes_rep = split_votes_df_by_party(df_votes, df_senators)
	
	cos_mat_dem = create_cosine_similarity_matrix(df_votes_dem, dissent_index=dissent_index)
	cos_mat_rep = create_cosine_similarity_matrix(df_votes_rep, dissent_index=dissent_index)
	
	return df_senators, df_votes_dem, df_votes_rep, cos_mat_dem, cos_mat_rep


def plot_individual_similarities_by_party(
	df_senators: pd.DataFrame,
	individual_similarities_dem: np.ndarray,
	individual_similarities_rep: np.ndarray,
	saveout: bool = False,
	saveout_fp: str = './individual_similarities_by_party.png',
	figsize: Tuple[float, float] = (30, 30)
) -> None:

	plt.figure(figsize=figsize)
	plt.boxplot([individual_similarities_dem, individual_similarities_rep])

	plt.xticks([1, 2], ['D', 'R'])
	plt.title('Distributions of Individual Cosine Similarities by Party')
	plt.xlabel('Party')
	plt.ylabel('Cosine Similarity')

	if saveout:
		plt.savefig(saveout_fp)

	plt.show()


if __name__ == '__main__':
	cos_sim(saveout_plot=True)