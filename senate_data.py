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

	return df_votes


def create_dataframes():

	return create_senators_df(), create_votes_df()

if __name__ == '__main__':
	create_dataframes()