import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from framework.Experiment import *
import json
import datetime

def run_experiment(dag):
	with open(os.path.join(os.environ['DAGS_PATH'], dag + '.json'),'r') as f:
		params = json.load(f)
	params['tag'] = dag
	print("Experiment {} started at {}".format(dag, datetime.datetime.now()))
	experiment = Experiment(params)
	experiment.run()
	print("Experiment {} finished at {}".format(dag, datetime.datetime.now()))

if __name__ == "__main__":
	scriptname = sys.argv[1]
	run_experiment(scriptname)
