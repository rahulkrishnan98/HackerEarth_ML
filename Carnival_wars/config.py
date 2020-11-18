import os
from datetime import datetime

#path creation --save_dir
input_tr_file = 'dataset/train.csv'
input_test_file = 'dataset/test.csv'
#-->test
test_path = "/Users/rahul/Documents/Projects/HackerEarth/Linear_regression/2020-11-17"
#-->train
time_stamp= datetime.utcnow().strftime("%Y-%m-%d") 
base_path = "/Users/rahul/Documents/Projects/HackerEarth/Linear_regression/"+time_stamp
if not os.path.exists(base_path):
    os.mkdir(base_path)

#Data IO
transform_cols = [
    'Loyalty_customer',
    'Product_Category'
]
ignore_cols= [
    'Product_id',
    'instock_date',
    'Customer_name'
]

models = [
    'Child_care',
    'Cosmetics',
    'Educational',
    'Fashion',
    'Home_decor',
    'Hospitality',
    'Organic',
    'Pet_care',
    'Repair',
    'Technology'
]
target_col = 'Selling_Price'