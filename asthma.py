import numpy as np
import re
from sys import argv

def getAsthma(filename):
	SIGN_PLUS = 1
	SIGN_MINUS = 0

	fcss_unique_data = []
	fcss_indexes = []
	sign_value = SIGN_PLUS # default

	with open(filename) as f:
		for line in f:
			str = line.strip()
			if len(str) > 0:
				re_sign = re.compile('^(plus)|(minus)')
				if re_sign.match(str):
					sign = re_sign.match(str).group()
					if sign == 'plus':
						sign_value = SIGN_PLUS
					else:
						sign_value = SIGN_MINUS
				else:
					re_split = re.compile(r' +')
					fcss_values = re_split.split(str)

					for fcss_value in fcss_values:
						if not fcss_value in fcss_unique_data:
							fcss_unique_data.insert(len(fcss_unique_data), fcss_value)

					fcss_indexes_count = len(fcss_indexes)
					fcss_indexes.insert(fcss_indexes_count, [fcss_unique_data.index(fcss_value) if fcss_value in fcss_unique_data  else None for fcss_value in fcss_values ])
					current_row = fcss_indexes[fcss_indexes_count]
					current_row.insert(len(current_row), sign_value)

	result = []
	fcss_dict = {}

	for row in fcss_indexes:
		current_result_row = [0]*(len(fcss_unique_data))
		for cell_index in range(len(row) - 1):
			fcss_dict[row[cell_index]] = fcss_dict.get(row[cell_index], 0) + 1
			current_result_row[cell_index] = fcss_dict[row[cell_index]]

		current_result_row[len(current_result_row) - 1] = row[len(row) - 1]

		result.insert(len(result), current_result_row)
	
	return np.asarray(result)