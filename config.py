import os
base_path = "/Users/rahul/Documents/Projects/HackerEarth"
input_tr_file = 'dataset/train.csv'

transform_cols = [
    'Loyalty_customer',
    'Product_Category'
]
ignore_cols = [
    'Product_id',
    'instock_date',
    'Customer_name'
]

target_col = 'Selling_Price'