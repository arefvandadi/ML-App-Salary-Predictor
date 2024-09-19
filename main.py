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
data_df.loc[(data_df["YearsCodePro"]=="Less than 1 year"), "YearsCodePro"] = 0.5
data_df.loc[(data_df["YearsCodePro"]=="More than 50 years"), "YearsCodePro"] = 50
data_df["YearsCodePro"] = data_df["YearsCodePro"].astype(float)
data_df["YearsCodePro"].unique()


######## Clean up the EdLevel column  ###########
# check the unique values
data_df["EdLevel"].unique()
data_df["EdLevel"].value_counts()


# Let's use a function and Apply function to clean up this column
def clean_education_level(x: str) -> str:
    """
    Helper function to clean up the EdLevel column
    """
    if "Master" in x:
        return "Master's degree"
    if "Bachelor" in x:
        return "Bachelor's degree"
    if "doctoral degree" in x or "Professional degree" in x:
        return "Post grad"
    return "Less than a Bachelors"

data_df["EdLevel"] = data_df["EdLevel"].apply(clean_education_level) 
data_df["EdLevel"].unique()

# Cleaning is almost done.
# Let's check how the data look like
data_df.head()
data_df.info()

data_df["YearsCodePro"] = data_df["YearsCodePro"].astype(float)
data_df.info()


########## Encoding the String Columns  #############
from sklearn.preprocessing import LabelEncoder

le_Country = LabelEncoder()
le_EdLevel = LabelEncoder()

data_df["Country"] = le_Country.fit_transform(data_df["Country"])
data_df["Country"].unique()

data_df["EdLevel"] = le_EdLevel.fit_transform(data_df["EdLevel"])
data_df["EdLevel"].unique()

# Check the Final data after All the Cleaning and Label encoding
data_df.info()





