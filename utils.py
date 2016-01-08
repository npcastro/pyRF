# coding=utf-8
# Archivo con métodos generales que ocupo repetidamente en mi codigo

# -----------------------------------------------------------------------------

def remove_duplicate_index(df):
    aux = df.index.value_counts() == 1
    b = aux[aux]
    df = df.loc[b.index]
    return df