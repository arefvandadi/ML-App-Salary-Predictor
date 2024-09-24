import streamlit as st
from page_options.predict_page import show_predict_page
from page_options.explore_page import show_explore_page

def main():
    st.sidebar.title('Predict/Explore')
    page = st.sidebar.selectbox("", ("Predict", "Explore"))

    if page == "Predict":
        show_predict_page()
    else:
        show_explore_page()


if __name__ == "__main__":
    main()