import pandas as pd

url_data = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
titanic_df = pd.read_csv(url_data, low_memory=False)  # no parsing of mixed types
