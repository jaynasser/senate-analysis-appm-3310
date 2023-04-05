from senate_data import create_dataframes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def plot_cosine_similarity_matrix(df_senators, df_votes, matrix, saveout=False, figsize=(30, 30), cmap='hot'):
	
	icpsr_to_name = dict(zip(df_senators['icpsr'], df_senators['bioname']))
	plot_labels = np.vectorize(icpsr_to_name.get)(list(df_votes.groupby('icpsr').groups))

	fig = plt.figure(figsize=figsize)
	ax = plt.gca()

	im = ax.matshow(matrix, cmap=cmap)
	fig.colorbar(im, fraction=0.046, pad=0.04)

	ax.set_xticks(np.arange(len(plot_labels)))
	ax.set_xticklabels(plot_labels, rotation=90)
	ax.set_yticks(np.arange(len(plot_labels)))
	ax.set_yticklabels(plot_labels)

	if saveout:
		plt.savefig('senate_cos_sim.png')

	plt.show()


def cos_sim():

	df_senators, df_votes = create_dataframes()
	cos_mat = create_cosine_similarity_matrix(df_votes)
	plot_cosine_similarity_matrix(df_senators, df_votes, cos_mat)


if __name__ == '__main__':
	cos_sim()