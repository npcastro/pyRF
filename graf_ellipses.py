import pickle
import matplotlib.pyplot as plt
import tree

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
from sklearn import cross_validation

path = 'sets/macho random.csv'
data = pd.read_csv(path)

# Elijo solo algunas clases

clases = [ 'Be Stars', 'MicroLensing', 'None Variable']
criterion = data['class'].map(lambda x: x in clases)
data = data[criterion]

# Para sacar una muestra estratificada de elementos
skf = cross_validation.StratifiedKFold(data['class'], n_folds=10)

for train_index, test_index in skf:
	data = data.iloc[test_index]
	break

# Elijo dos variables
#x = 'B-R_R'
x = 'Period_SNR_R'
y = 'Eta_R'

# obtengo los valores de las elipses
x_mean = data[x + '.mean'].tolist()
x_margin = (data[x + '.r'] - data[x + '.l']).tolist()

y_mean = data[y + '.mean'].tolist()
y_margin = (data[y + '.r'] - data[y + '.l']).tolist()

colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }
class_list = data['class'].tolist()

ells = []
for i in xrange(len(x_mean)):
	ells.append( Ellipse(xy=[x_mean[i], y_mean[i]], width=x_mean[i], height=y_mean[i], angle=0, facecolor=colors[class_list[i]]) )

fig = plt.figure()
ax = fig.add_subplot(111)

plt.xlim(-0.1,0.8)
plt.ylim(-0.1,1.5)

plt.xlabel(x)
plt.ylabel(y, rotation = 'horizontal')

plt.axhline(y = 0.66, color = 'r')
# plt.axvline(x = 0.38, color = 'r')
plt.axvline(x = 0.07, color = 'r')

for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)

plt.show()
plt.close()
