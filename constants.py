import os

from dotenv import load_dotenv

load_dotenv()

SENATORS_CSV_FP = os.environ['SENATORS_CSV_FP']
VOTES_CSV_FP = os.environ['VOTES_CSV_FP']