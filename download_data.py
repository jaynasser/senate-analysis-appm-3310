from constants import VOTEVIEW_DATA_URL

import os
from typing import Iterable, Literal

import requests


def download_csvs(chamber: Literal['S', 'H'], congresses: Iterable[int] = range(70, 119)):

	for congress in congresses:
		congress_str = str(congress).zfill(3)
		request_strs = (
			VOTEVIEW_DATA_URL + f'/members/{chamber}{congress_str}_members.csv',
			VOTEVIEW_DATA_URL + f'/votes/{chamber}{congress_str}_votes.csv'
		)

		for request_str in request_strs:
			output_file = './data/' + request_str.split('/')[-1]
			if not os.path.isfile(output_file):
				re = requests.get(request_str)
				with open(output_file, 'wb+') as f:
					f.write(re.content)


if __name__ == '__main__':
	download_csvs(chamber='S')