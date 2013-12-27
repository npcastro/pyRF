import tree
import numpy as np

if __name__ == '__main__':
    #nombres = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    #data = pd.read_csv('iris.data', header=None, names=nombres)

    # nombres = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class', 'sepal length_conf', 'sepal width_conf', 'petal length_conf', 'petal width_conf']
    # data = pd.read_csv('iris_comp.data', header=None, names=nombres)

    path = "Resultados/Resultados 40.txt"
    nombres = ['Macho_id', 'Sigma_B', 'Sigma_B_conf', 'Eta_B', 'Eta_B_conf', 'stetson_L_B', 'stetson_L_B_conf',
               'CuSum_B', 'CuSum_B_conf', 'B-R', 'B-R_conf', 'class']
    data = tree.pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)
    data = data.reindex(np.random.permutation(data.index))

    train = tree.pd.concat([data[0:98], data[105:197], data[204:310]])

    test = tree.pd.concat([data[98:105], data[197:204], data[310:]])

    #clf = Tree('gain')
    clf = tree.Tree('confianza')
    clf.fit(train)

    result = clf.predict_table(test)
    matrix = clf.confusion_matrix(result)