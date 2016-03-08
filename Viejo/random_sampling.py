# Toma un dataset con incertidumbre y genera n datasets nuevos sampleados de el. 

import pandas as pd
from numpy import random
import os

def filterfeatures(df):
	"""Toma un dataframe y retorna el nombre de las features.
	"""
	filter_arr = []
	for f in df.columns:
		if not '.l' in f and not '.r' in f and not '.std' in f and f != 'weight' and f != 'class':
			# filter_arr.append(f.rstrip('.mean'))
			filter_arr.append(f)
	return filter_arr

def sample_row(row, columns):
	"""Recibe una fila de datos con incertidumbre gaussiana y retorna un dato sampleado aleatoriamente de esta.
	"""
	sampled_row = pd.Series( index = columns)

	# Sampleo cada feature segun la distribucion de la fila
	for c in columns:
		c = c.rstrip('.mean')
		sampled_row[c + '.mean'] = random.normal(row[c + '.mean'], row[c + '.std'])

	# Agrego la columna clase
	sampled_row['class'] = row['class']

	return sampled_row

def save(df, path):
    """Save a figure from pyplot.
 
    Parameters
    ----------
    path : string
        The path (and filename) to save the
        figure to.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'
 
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # The final path to save to
    savepath = os.path.join(directory, filename)
 
    # Save the dataset
    sampled_frame.to_csv(savepath, index=False) 


if __name__ == '__main__':
	aux = [2, 6, 11, 16, 21, 26, 31, 36, 41]
	for u in aux:
	
		# path = 'sets/macho random sampled 10 folds.csv'
		path = 'sets/macho %/macho random ' + str(u) +'.csv'
		data = pd.read_csv(path)
		
		columns = filterfeatures(data)

		n = 15
		for i in xrange(n):

			sampled_frame = data.apply(sample_row, axis=1, args=[columns])
			save(sampled_frame, 'sets/sampling/' + str(u - 1) + ' %/sampled macho ' + str(u - 1) + '% ' + str(i) + '.csv')
			# sampled_frame.to_csv('sets/sampling/sampled macho ' + str(i) + '.csv', index=False) 
