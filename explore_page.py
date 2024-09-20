import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATAPOINT_CUTOFF = 400

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


@st.cache_data
def load_data():
    data_df = pd.read_csv("./data/stack-overflow-developer-survey-2020/survey_results_public.csv")
    data_df = data_df[["Country", 'EdLevel', "YearsCodePro", 'Employment','ConvertedComp']]
    data_df.dropna(inplace=True)
    data_df.reset_index(inplace=True, drop=True)
    data_df.rename(columns={"ConvertedComp":"Salary"}, inplace=True)
    data_df = data_df[data_df["Employment"]=="Employed full-time"]
    data_df.drop("Employment", axis=1, inplace=True)
    data_df = filter_countries_per_occurence(data_df, DATAPOINT_CUTOFF)
    data_df = data_df[(data_df["Salary"]<250000) & (data_df["Salary"]>10000)]
    data_df.loc[(data_df["YearsCodePro"]=="Less than 1 year"), "YearsCodePro"] = 0.5
    data_df.loc[(data_df["YearsCodePro"]=="More than 50 years"), "YearsCodePro"] = 50
    data_df["YearsCodePro"] = data_df["YearsCodePro"].astype(float)
    data_df["EdLevel"] = data_df["EdLevel"].apply(clean_education_level)
    return data_df

data_df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
    ### Stack Overflow Developer Survey 2020
    """
    )

    data = data_df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)


    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = data_df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)


    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = data_df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)




