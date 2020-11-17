from sklearn import preprocessing

def label_transform(df, col):
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df, le

