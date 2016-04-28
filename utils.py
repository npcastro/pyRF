# coding=utf-8
# Archivo con métodos generales que ocupo repetidamente en mi codigo

# -----------------------------------------------------------------------------

from sklearn import cross_validation

def remove_duplicate_index(df):
    aux = df.index.value_counts() == 1
    b = aux[aux]
    df = df.loc[b.index]
    return df

def equalize_indexes(df1, df2):
    df1 = remove_duplicate_index(df1)
    df2 = remove_duplicate_index(df2)
    common_index = list(set(df1.index.tolist()) & set(df2.index.tolist()))
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    df2 = df2.sort_index()
    df1 = df1.sort_index()

    return df1, df2

def stratified_filter(df, y, percentage=0.1):
    """ Toma un dataframe. Y retorna un porcentaje de las tuplas, manteniendo la proporción,
    de elementos encontrados en y.
    """

    skf = cross_validation.StratifiedKFold(y, n_folds=int(1 / percentage))
    aux = [x for x in skf]
    return df.iloc[aux[0][1]]

def filter_data(df, index_filter=None, class_filter=None, feature_filter=None, lc_filter=None):
    """ Filtra segun indices, porcentaje de curvas, clases, features y por ultimo separa
    las variables de la clase
    """
    if index_filter is not None:
        df = df.loc[index_filter]
    elif lc_filter is not None:
        df = stratified_filter(df, df['class'], lc_filter)
    
    if class_filter:
        df = df[df['class'].apply(lambda x: True if x in class_filter else False)]

    df = df.dropna(axis=0, how='any')
    y = df['class']
    df = df.drop('class', axis=1)

    if feature_filter:
        df = df[feature_filter]

    return df, y