# coding=utf-8

# Script solo pa recordar como separe los sets de testing para eros y macho. No esta 
# pensado para correrse normalmente

# -----------------------------------------------------------------------------

from sklearn import cross_validation
import pandas as pd

import utils

catalog = 'EROS'
test_size = 5000

normal_df = pd.read_csv('/n/home09/ncastro/workspace/Features/sets/' + catalog + '/' + catalog + '_regular_set_5.csv', index_col=0)
sample_df = pd.read_csv('/n/seasfs03/IACS/TSC/ncastro/sets/' + catalog + '_Sampled/uniform/5%/' + catalog + '_sampled_0.csv', index_col=0)

a, b = utils.equalize_indexes(normal_df, sample_df)

sss = cross_validation.StratifiedShuffleSplit(a['class'], n_iter=1, test_size=test_size, 
											  random_state=1)

for train_index, test_index in sss:
	train_df = a.iloc[train_index]
	test_df = a.iloc[test_index]

a.to_csv('/n/home09/ncastro/workspace/Features/sets/Common/' + catalog + '.csv')
test_df.to_csv('/n/home09/ncastro/workspace/Features/sets/Common/' + catalog + '_test.csv')
