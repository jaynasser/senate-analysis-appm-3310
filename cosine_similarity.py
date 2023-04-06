from senate_data import create_senators_df, create_votes_df

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_votes_df_by_party(df_votes, df_senators):

	icpsr_to_party = dict(zip(df_senators['icpsr'], df_senators['party_code']))

	df_votes_party = df_votes.copy()
	df_votes_party['party_code'] = df_votes.icpsr.map(icpsr_to_party)

	df_votes_dem = df_votes_party[df_votes_party['party_code'] == 100]
	df_votes_rep = df_votes_party[df_votes_party['party_code'] == 200]

	return df_votes_dem, df_votes_rep


def create_vote_matrix(df_votes):
	
	senator_vote_array = df_votes.to_numpy(dtype='int')

	num_rollcalls = len(df_votes.groupby('rollnumber')) # number of votes that occurred

	vote_matrix = np.reshape(senator_vote_array[: , 2], (num_rollcalls, -1))

	return vote_matrix


def create_cosine_similarity_matrix(df_votes):

	vote_matrix = create_vote_matrix(df_votes)

	d = vote_matrix.T @ vote_matrix
	norm = (vote_matrix * vote_matrix).sum(0, keepdims=True) ** .5
	cos_mat = d / norm / norm.T
	
	return cos_mat


def plot_cosine_similarity_matrix(df_senators, df_votes, matrix, saveout=False, saveout_fp='senate_cos_sim.png', figsize=(30, 30), cmap='hot'):
	
	icpsr_to_name = dict(zip(df_senators['icpsr'], df_senators['bioname']))
	plot_labels = np.vectorize(icpsr_to_name.get)(list(df_votes.groupby('icpsr').groups))

	fig = plt.figure(figsize=figsize)
	ax = plt.gca()

	im = ax.matshow(matrix, cmap=cmap)
	fig.colorbar(im, fraction=0.046, pad=0.04)
	im.set_clim(0, 1)

	ax.set_xticks(np.arange(len(plot_labels)))
	ax.set_xticklabels(plot_labels, rotation=90)
	ax.set_yticks(np.arange(len(plot_labels)))
	ax.set_yticklabels(plot_labels)

	if saveout:
		plt.savefig(saveout_fp)

	plt.show()


def cos_sim(df_senators=None, df_votes=None, saveout_plot=False, saveout_fp=['senate_cos_sim_dem.png', 'senate_cos_sim_rep.png']):

	if df_senators is None:
		df_senators = create_senators_df()

	if df_votes is None:
		df_votes = create_votes_df()
	
	df_votes_dem, df_votes_rep = split_votes_df_by_party(df_votes, df_senators)
	
	cos_mat_dem, cos_mat_rep = create_cosine_similarity_matrix(df_votes_dem), create_cosine_similarity_matrix(df_votes_rep)
	plot_cosine_similarity_matrix(df_senators, df_votes_dem, cos_mat_dem, saveout=saveout_plot, saveout_fp=saveout_fp[0])
	plot_cosine_similarity_matrix(df_senators, df_votes_rep, cos_mat_rep, saveout=saveout_plot, saveout_fp=saveout_fp[1])


if __name__ == '__main__':
	cos_sim(saveout_plot=True)