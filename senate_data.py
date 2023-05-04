from constants import SENATORS_CSV_FP, VOTES_CSV_FP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_icpsrs_to_ignore(df_senators: pd.DataFrame) -> None:

	global icpsrs_to_ignore
	icpsrs_to_ignore = list(
		df_senators[(df_senators['chamber'] == 'President') | (~df_senators['party_code'].isin([100, 200]))]['icpsr']
	)


def create_senators_df() -> pd.DataFrame:

	df_senators = pd.read_csv(SENATORS_CSV_FP)

	if 'icpsrs_to_ignore' not in globals():
		get_icpsrs_to_ignore(df_senators)

	df_senators.drop(df_senators[df_senators.icpsr.isin(icpsrs_to_ignore)].index, inplace=True)
	df_senators = df_senators[['icpsr', 'state_abbrev', 'party_code', 'bioname']]

	return df_senators


def create_votes_df() -> pd.DataFrame:

	df_votes = pd.read_csv(VOTES_CSV_FP)

	df_votes.drop(['congress', 'chamber', 'prob'], axis=1, inplace=True)

	if 'icpsrs_to_ignore' not in globals():
		raise NameError('Global variable icpsrs_to_ignore must be defined. \
			Run get_icpsrs_to_ignore() or create_senators_df() first to fix this.')

	df_votes.drop(df_votes[df_votes.icpsr.isin(icpsrs_to_ignore)].index, inplace=True)

	df_votes['cast_code'] = df_votes['cast_code'].map(
		{1: 1, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: 0, 8:0, 9: 0} # reassign cast codes (1 for yea, -1 for nea, 0 for present or abstain)
	)

	present_combinations = set(df_votes.groupby(['rollnumber', 'icpsr']).groups)
	all_combinations = set()

	for icpsr in df_votes['icpsr'].unique():
		for rollnumber in df_votes['rollnumber'].unique():
			all_combinations.add((rollnumber, icpsr))
	
	if (all_combinations - present_combinations):
		rollnumbers_to_add, icpsrs_to_add = list(zip(*list(all_combinations - present_combinations)))
	
		data_to_add = pd.DataFrame({
			'icpsr': icpsrs_to_add,
			'rollnumber': rollnumbers_to_add,
			'cast_code': [0 for _ in range(len(rollnumbers_to_add))]
		})
	
		df_votes = pd.concat((df_votes, data_to_add)).sort_values(by=['rollnumber', 'icpsr'])
	
	senator_groupby = df_votes.groupby('icpsr')['cast_code']
	relative_zeros = (senator_groupby.apply(lambda x: (x == 0).sum()) / senator_groupby.size()).rename('relative_zeros')
	df_votes = df_votes.merge(relative_zeros, how='left', on='icpsr')

	df_votes = df_votes[df_votes['relative_zeros'] < 0.25]
	df_votes.drop(columns='relative_zeros')

	return df_votes