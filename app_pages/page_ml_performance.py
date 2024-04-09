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

    st.write("### Model History")

    col1, col2 = st.beta_columns(2)
    with col1: 
        model_accuracy = plt.imread(f"streamlit_images/model_accuracy.png")
        st.image(model_accuracy, caption='Training Accuracy of the Model')
    with col2:
        model_loss = plt.imread(f"streamlit_images/model_loss.png")
        st.image(model_loss, caption='Training Loss of the Model')

    st.warning(
        f"**Training Accuracy/Loss of the Model**\n\n"
        f"The graphs showing the training accuracy and loss of the model "
        f"were obtained by using SoftMax as the activation function on our "
        f"model. It can be observed that in both accuracy and loss graphs "
        f"that there is slight overfitting in the initial epochs and in "
        f"later epochs we can observe even more overfitting."
    )

    confusion_matrix = plt.imread(f"")
    st.image(
        confusion_matrix, 
        caption='Confusion Matrix'
    )

    st.warning(
        f"**Confusion Matrix**"
        f"A confusion matrix summerises the performance of a machine learning "
        f"model on a set of test data. It is a way of displaying the number "
        f"of accurate and inaccurate instances based on the predictions of "
        f"the model. Also, it measures the performance of classification "
        f"models, which aims to predict a categorical label for each input.\n\n"
        f"A confusion matrix displays the number of instances produced by "
        f"the model on the test data it is given.\n\n"
        f"* True Positives (TP): This occurs when the model accurately "
        f"predicts a positive data point.\n"
        f"* True Negatives (TN): This occurs when the model accurately "
        f"predicts a negative data point.\n"
        f"* False Positives (FP): This occurs when the model predicts a "
        f"positive data point incorrectly.\n"
        f"* False Negatives (FN): This occurs when the model mispredicts a "
        f"negative data point.\n\n"
        f"A confusion matrix is essential when it comes to assessing a "
        f"classification model's performance. It gives a thorough analysis "
        f"of true positive, true negative, false positive and false negative "
        f"predictions, thereby facilitating a comprehension of a model's "
        f"recall, accuracy, precision and overall effectiveness in class "
        f"distinction."
    )

    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))

    st.write(
        f"For more information, please visit and read the "
        f"[Project README file](https://github.com/lgaudencio/brain-tumor-detection)"
    )