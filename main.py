from sklearn import linear_model
from sklearn import model_selection
import pandas as pd
import os
import pickle
import joblib
import config
import utils


if __name__=='__main__': 
    df = pd.read_csv(
        config.input_tr_file,
        error_bad_lines = True
    ).dropna()
    df = df.drop(labels= config.ignore_cols, axis=1)
    meta = {}
    for col in config.transform_cols:
        df, le = utils.label_transform(
            df,
            col
        )
        meta[col]= le
    Category_dict = {
        meta['Product_Category'].transform([key])[0] : key
        for key in meta['Product_Category'].classes_
    }
    joblib.dump(meta, os.path.join(config.base_path,'meta.bin'))
       

    train, test = model_selection.train_test_split(
        df,
        test_size = 0.2,
        stratify = df.Product_Category
    )

    for _, category in enumerate(set(df.Product_Category.to_list())):
        category_name = Category_dict[category]
        category_model = f'{category_name}_model.pkl'

        target_train = train[train.Product_Category==category]
        target_test = test[test.Product_Category==category]

        X_train = target_train[[col for col in target_train.columns if col != config.target_col]]
        X_test = target_test[[col for col in target_test.columns if col != config.target_col]]
        y_train = target_train[config.target_col]
        y_test = target_test[config.target_col]

        model = linear_model.LinearRegression(
            normalize=True,
        ).fit(
            X= X_train, 
            y= y_train
        )
        score = model.score(
            X= X_test,
            y= y_test
        )
        print(f'Score for {category_name} is {score}')
        dir_ = os.path.join(config.base_path,category_name)
        os.mkdir(dir_)
        pickle.dump(model, open(os.path.join(
            dir_, category_model
        ), 'wb'))



