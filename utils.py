# coding=utf-8
# Archivo con métodos generales que ocupo repetidamente en mi codigo

# -----------------------------------------------------------------------------

def remove_duplicate_index(df):
    aux = df.index.value_counts() == 1
    b = aux[aux]
    df = df.loc[b.index]
    return df

def filter_data(df, index_filter=None, class_filter=None, feature_filter=None):
    if index_filter is not None:
        df = df.loc[index_filter]
    
    if class_filter:
        df = df[df['class'].apply(lambda x: True if x in class_filter else False)]

    df = df.dropna(axis=0, how='any')
    y = df['class']
    df = df.drop('class', axis=1)

    if feature_filter:
        df = df[feature_filter]

    return df, y

def equalize_indexes(df1, df2):
    common_index = list(set(df1.index.tolist()) & set(df2.index.tolist()))
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    df2 = df2.sort_index()
    df1 = df1.sort_index()

    return df1, df2
