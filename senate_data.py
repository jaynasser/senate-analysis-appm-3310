from constants import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_senators_df():

	df_senators = pd.read_csv(SENATORS_CSV_FP)

	df_senators.party_code.replace(328, 100, inplace=True) # classify independents as democrats for simplicity
	df_senators.drop(df_senators[df_senators.icpsr.isin(ICPSRS_TO_IGNORE)].index, inplace=True)
	df_senators = df_senators[['icpsr', 'state_abbrev', 'party_code', 'bioname']]

	return df_senators


def create_votes_df():

	df_votes = pd.read_csv(VOTES_CSV_FP)

	df_votes.drop(['congress', 'chamber', 'prob'], axis=1, inplace=True)
	df_votes.drop(df_votes[df_votes.icpsr.isin(ICPSRS_TO_IGNORE)].index, inplace=True)

	df_votes['cast_code'] = df_votes['cast_code'].map({1: 1, 6: -1, 7: 0, 9: 0}) # reassign cast codes (1 for yea, -1 for nea, 0 for present or abstain)
	
	present_combinations = set(df_votes.groupby(['rollnumber', 'icpsr']).groups)
	all_combinations = set()
	
	for icpsr in df_votes['icpsr'].unique():
		for rollnumber in df_votes['rollnumber'].unique():
			all_combinations.add((rollnumber, icpsr))
			
	rollnumbers_to_add, icpsrs_to_add = list(zip(*list(all_combinations - present_combinations)))
	
	data_to_add = pd.DataFrame({
		'icpsr': icpsrs_to_add,
		'rollnumber': rollnumbers_to_add,
		'cast_code': [0 for _ in range(len(rollnumbers_to_add))]
	})
	
	df_votes = pd.concat((df_votes, data_to_add))
	
	return df_votes


if __name__ == '__main__':
	df_senators, df_votes = create_senators_df(), create_votes_df()