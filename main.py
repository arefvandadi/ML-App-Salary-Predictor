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

# let's only focus on full-time employment 
# and we can drop the employment column after
data_df = data_df[data_df["Employment"]=="Employed full-time"]
data_df.head(10)
data_df.drop("Employment", axis=1, inplace=True)

# Data Per Country
data_df["Country"].value_counts()
data_df["Country"].value_counts().index
data_df["Country"].value_counts().values

# Defined a function to remove all countries with less data than a cutoff value
def filter_countries_per_occurence(data_df:pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """
    Removes countries with number of data points below cutoff
    Parameters:
        cutoff (int): number of datapoint for countries.
    """
    selected_countries = [data_df["Country"].value_counts().index[i] for i, j in enumerate(data_df["Country"].value_counts().values) if j>cutoff]
    data_df = data_df[data_df["Country"].isin(selected_countries)]
    return data_df

# Use the above function to remove all countries with smaller than 400 datapoints.
data_df = filter_countries_per_occurence(data_df, 400)

#Check if countries with low datapoints were removed
data_df["Country"].value_counts()


# BoxPlot of Salaries per Countries
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
data_df.boxplot("Salary", "Country", ax = ax)
plt.title("Salary ($US) vs Country")
plt.suptitle("")
plt.ylabel("Salary")
plt.xticks(rotation=90)
plt.show()

# Removing Outliers
# Looking at the plot, Most of the data sit between 0 to 250k.
# Let's remove the outliers: >250k and <10k
data_df = data_df[(data_df["Salary"]<250000) & (data_df["Salary"]>10000)]

#Check min and max to souble check if outliers were removed
data_df["Salary"].min()
data_df["Salary"].max()

# BoxPlot of Salaries per Countries After Ouliers removed
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
data_df.boxplot("Salary", "Country", ax = ax)
plt.title("Salary ($US) vs Country")
plt.suptitle("")
plt.ylabel("Salary")
plt.xticks(rotation=90)
plt.show()

######## Clean up the YearsCodePro column  ###########
# check the unique values
data_df["YearsCodePro"].unique()

# There are "Less than 1 Year" and "More than 50 years" values
# Setting thee two to 0.5 and 50 respectively
data_df[data_df["YearsCodePro"]=="Less than 1 year"] = 0.5
data_df[data_df["YearsCodePro"]=="More than 50 years"] = 50
data_df["YearsCodePro"].unique()

