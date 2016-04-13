# coding=utf-8
# Entreno un modelo (random forest) sobre un set de entrenamiento normal
# Después al testear testeo con las medias del GP, en lugar de con las feats normales

# -------------------------------------------------------------------------------------------------

import argparse
import pickle
import sys

import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import metrics

print ' '.join(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--percentage', required=True, type=str)
parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
parser.add_argument('--folds',  required=True, type=int)
parser.add_argument('--regular_set_path',  required=True, type=str)
parser.add_argument('--mean_set_path',  required=True, type=str)
parser.add_argument('--result_path',  required=True, type=str)
parser.add_argument('--feature_filter',  nargs='*', type=str)

args = parser.parse_args(sys.argv[1:])

percentage = args.percentage
catalog = args.catalog
folds = args.folds
regular_set_path = args.regular_set_path
mean_set_path = args.mean_set_path
result_path = args.result_path
feature_filter = args.feature_filter	

# Necesito asgurarme de tener los mismos ids en ambos sets. Normal y con medias
train_data = pd.read_csv(regular_set_path, index_col=0)
test_data = pd.read_csv(mean_set_path, index_col=0)

# Elimino indices repetidos
train_data = train_data.groupby(train_data.index).first()
test_data = test_data.groupby(test_data.index).first()

# Me aseguro de clasificar con las mismas curvas
train_data = train_data.loc[test_data.index]
test_data = test_data.loc[train_data.index]

# Sorteo ambos sets para que esten en el mismo orden
train_data = train_data.sort()
test_data = test_data.sort()

# Separo features de las clases
train_y = train_data['class']
train_X = train_data.drop('class', axis=1)

test_y = test_data['class']
test_X = test_data.drop('class', axis=1)

if feature_filter:
	train_X = train_X[feature_filter]
	test_X = test_X[feature_filter]

skf = cross_validation.StratifiedKFold(train_y, n_folds=folds)

results = []
ids = []

for train_index, test_index in skf:
    fold_train_X = train_X.iloc[train_index]
    fold_train_y = train_y.iloc[train_index]

    fold_test_X = test_X.iloc[test_index]
    fold_test_y = test_y.iloc[test_index]

    clf = None
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                     max_depth=14, min_samples_split=5,
                                     n_jobs=-1)

    clf.fit(fold_train_X, fold_train_y)
    results.append(metrics.predict_table(clf, fold_test_X, fold_test_y))
    ids.extend(fold_test_X.index.tolist())

result = pd.concat(results)
result['indice'] = ids
result.set_index('indice')
result.index.name = catalog + '_id'
result = result.drop('indice', axis=1)

output = open(result_path + 'Arboles/Arbol_' + percentage + '.pkl', 'wb+')
pickle.dump(clf, output)
output.close()

result.to_csv(result_path + 'Predicciones/result_' + percentage + '.csv')
