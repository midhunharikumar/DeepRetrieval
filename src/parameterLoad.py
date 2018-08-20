import json
import numpy as np

parameters_filename = 'parameters.json'


def train_parameter_load():
	json_file=open('parameters.json','r').read()
	parameters = json.loads(json_file)
	return parameters
