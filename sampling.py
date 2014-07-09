import pandas as pd
import math

from sklearn import cross_validation


from config import *

def sampling_data(path, fold=10):
  data = pd.read_csv(path)

  skf = cross_validation.StratifiedKFold(data['class'], n_folds=fold)

  for train_index, test_index in skf:
    data = data.iloc[test_index]
    break

  name = path.split('/')[-1].split('.')[0]
  data.to_csv(SETS_PATH + "/" + name + " sampled " + str(percent) + " folds.csv", index=False)