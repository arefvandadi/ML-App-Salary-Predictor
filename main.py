import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv("./data/stack-overflow-developer-survey-2020/survey_results_public.csv")

data_df.head(10)

# Number of data points
data_df.shape[0]

# Check the columns
data_df.columns

# select 5 columns and rop the rest 
data_df = data_df[["Country", 'EdLevel', "YearsCodePro", 'Employment','ConvertedComp']]

# check NAN in column
data_df.isna().sum()

# drop NaN entries and check the number of NaN after
data_df.dropna(inplace=True)
data_df.isna().sum()

# reset the index
data_df.reset_index(inplace=True)

# Change th column name from ConvertedComp to Salary
data_df.rename(columns={"ConvertedComp":"Salary"}, inplace=True)
data_df.head(10)


