import pandas as pd
import config
import utils
import joblib
import os


if __name__ == '__main__':
    models_dict = utils.load_models(
        path = config.test_path,
        models= config.models
    )
    meta = joblib.load(os.path.join(config.test_path,'meta.bin'))
    df = pd.read_csv(config.input_test_file)
    

    submit_csv = pd.DataFrame(columns=['Product_id','Selling_Price'])
    submit_csv['Product_id'] = df['Product_id']
    submit_csv['Selling_Price'] = pd.Series([0]*len(df))

    df = df.drop(labels=config.ignore_cols, axis=1)

    Category_dict = {
        meta['Product_Category'].transform([key])[0] : key
        for key in meta['Product_Category'].classes_
    }
    
    for cols in df.columns:
        df[cols] = df[cols].fillna(-1)
    #label encode transform
    for key, value in meta.items():
        df[key] = df[key].apply(lambda x: value.transform([x])[0])

    #model prediction
    for ind, row in df.iterrows():
        model = models_dict[Category_dict[row['Product_Category']]]
        pred = model.predict([row])
        submit_csv['Selling_Price'].iloc[ind] = max(pred, -1*pred)
    submit_csv.to_csv("submit.csv")