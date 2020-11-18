from sklearn import preprocessing
from tqdm import tqdm
import os
import pickle

def label_transform(df, col):
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df, le

def load_models(
    path,
    models
):
    loaded_models = {}
    for _, model in tqdm(enumerate(models)):
        filename = os.path.join(path,model,f'{model}_model.pkl')
        loaded_models[model] = pickle.load(open(filename, 'rb'))
    return loaded_models