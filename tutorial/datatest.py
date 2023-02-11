
import numpy as np
from typing import Union
import pandas as pd

# elapsed time
#saved times:
time = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
#saved inputs:
inputs = [{'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}, {'i': 2.0}]
#saved outputs
outputs = [{'t': 292.1, 'v': 4.1828702595530105}, {'t': 292.22644204150873, 'v': 3.950865081712307}, {'t': 292.3610888245951, 'v': 3.9288409870976757}, {'t': 292.49484966409074, 'v': 3.9203754582002106}, {'t': 292.6271089174722, 'v': 3.914547058110963}, {'t': 292.7577558424454, 'v': 3.9092570975639114}, {'t': 292.8867852787106, 'v': 3.904102121211562}, {'t': 293.0142124723636, 'v': 3.8990044036060914}, {'t': 293.14005640048686, 'v': 3.8939488081511775}, {'t': 293.2643365580705, 'v': 3.8889322262832393}, {'t': 293.3870723426943, 'v': 3.883953862519164}]
#saved event states
EOD = [{'EOD': 1.0000418413269898}, {'EOD': 0.9948974424148921}, {'EOD': 0.9897530581526326}, {'EOD': 0.9846086774416868}, {'EOD': 0.9794642984979828}, {'EOD': 0.9743199209705339}, {'EOD': 0.9691755447837648}, {'EOD': 0.9640311699150712}, {'EOD': 0.958886796352081}, {'EOD': 0.9537424240844369}, {'EOD': 0.9485980531022217}]


output = pd.DataFrame(time)
output.columns = ["time"]

new_column = pd.DataFrame.from_records(sav_inputs)  # creates new dataframe column
new_column_label = new_column.columns[0]   # gets column label
output[new_column_label] = new_column # merges new column to the right most spot of the dataframe

event_data = pd.DataFrame.from_records(saved_event_state)  # creates column for input data under the label 'EOD'
ed_column_title = event_data.columns
output[ed_column_title] = event_data

print(output)
#print(type(sav_inputs[0]))


key = "test"
value = 5


# self.matrix[self._keys.index(key)] = np.atleast_1d(value)

# list of vowels
a = np.matrix('1 2; 3 4')

# iter() with a list of vowels
vowels_iter = pd.DataFrame.from_records(a)
#vowels_iter.columns = vowels
print(vowels_iter)



