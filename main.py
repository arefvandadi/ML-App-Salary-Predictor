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
data_df.reset_index(inplace=True, drop=True)

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

# Define X and Y Training Data from the Original Data
X = data_df.drop("Salary", axis=1)
y = data_df["Salary"]


# Define Train Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

########## Linear Regression ###############
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np
error_mean_square = mean_squared_error(y_test, y_pred)
error = np.sqrt(error_mean_square).item()
print(f"Linear Regression Error: ${error:.2f}")


########## DecisionTreeRegressor ###############
from sklearn.tree import DecisionTreeRegressor
decision_tree_regresser = DecisionTreeRegressor(random_state=0)
decision_tree_regresser.fit(X_train, y_train)
y_pred = decision_tree_regresser.predict(X_test)

error_mean_square = mean_squared_error(y_test, y_pred)
error = np.sqrt(error_mean_square).item()
print(f"Decision Tree Regressor Error: ${error:.2f}")


########## Random Forest Regressor ##################
from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X_train, y_train)
y_pred = random_forest_reg.predict(X_test)

error_mean_square = mean_squared_error(y_test, y_pred)
error = np.sqrt(error_mean_square).item()
print(f"Random Forest Regressor Error: ${error:.2f}")



############### Grid Search Random Forst Regressor Grid Search ############
from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

RF_regressor = RandomForestRegressor(random_state=0)
RF_gs = GridSearchCV(RF_regressor, parameters, scoring='neg_mean_squared_error')
RF_gs.fit(X_train, y_train)

RF_regressor_best = RF_gs.best_estimator_

RF_regressor_best.fit(X_train, y_train)
y_pred = RF_regressor_best.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Best Random Forest Regressor Error: ${error:,.02f}")
# Random Forest Regressor Error: $33256.00
# Best Random Forest Regressor Error (Max Depth:8): $29,730.58


############### Grid Search Decision Tree Regressor Grid Search ############
max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

DecTree_regressor = DecisionTreeRegressor(random_state=0)
DT_gs = GridSearchCV(DecTree_regressor, parameters, scoring='neg_mean_squared_error')
DT_gs.fit(X_train, y_train)

DecTree_regressor_best = DT_gs.best_estimator_

DecTree_regressor_best.fit(X_train, y_train)
y_pred = DecTree_regressor_best.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Best Decision Tree Regressor Error: ${error:,.02f}")
# Decision Tree Regressor Error: $41927.26
# Best Decision Tree Regressor Error (Max Depth:86): $30,169.21


########## Let's Save the Best Model into "model"  ##########
model = RF_regressor_best

########## Let's test the Best Model with a Random Data  ##########
x_data = np.array([["United States", "Master's degree", 25]])

# Convert the data with label encoders and then to float
x_data[:, 0] = le_Country.fit_transform(x_data[:, 0])
x_data[:, 1] = le_EdLevel.fit_transform(x_data[:, 1])
x_data = x_data.astype(float)
y_data_pred = model.predict(x_data)
print(f"Testing the Best Model with random data --> y_pred = {y_data_pred}")





