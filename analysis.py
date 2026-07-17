import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/student_placement.csv")

st.title("📊 Student Placement Data Analysis")

st.write(df)

st.subheader("Placement Distribution")

df["placement_status"].value_counts().plot(
    kind="bar"
)

st.pyplot(plt)