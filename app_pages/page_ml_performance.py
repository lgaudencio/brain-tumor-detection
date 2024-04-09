import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.info(
        f"In this page, a presentation of how the dataset was divided, "
        f"the model performance on the data given and a brief explanation "
        f"of each result will be provided."
    )

    st.write("### Distribution of Dataset Images per Set and Label")

    bar_chart_labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(
        bar_chart_labels_distribution, 
        caption='Labels Distribution on Train, Validation and Test Sets'
    )

    pie_chart_labels_distribution = plt.imread(f"outputs/{version}/healthy_and_tumor_distribution.png")
    st.image(
        pie_chart_labels_distribution, 
        caption='Labels Distribution on MRI Scan Labels (Healthy & Tumor)'
    )

    st.warning(
        f"The brain MRI dataset was split into three different subsets, "
        f"these subsets were: Train, Validation and Test.\n\n"
        f"The train subset was made from 70% of the entire dataset and it "
        f"will be used my the model to learn on how to make predictions on "
        f"new unseen data.\n\n"
        f"The validation subset was made from 10% of the entire dataset and "
        f"it is used to improve the performance of the model by fine tuning "
        f"the model after each epoch.\n\n"
        f"The test subset was made from 20% of the entire dataset and it "
        f"will give us information about its final accuracy. The images in "
        f"this subset was never seen by the model while it was training. "
    )

    st.write("---")