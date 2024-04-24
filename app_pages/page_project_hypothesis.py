import streamlit as st
import matplotlib.pyplot as plt 


def page_project_hypothesis_body():
    st.write("### Hypothesis 1 and Validation")

    st.success(
        f"Brain scans which contain the presence of a tumour can be "
        f"visually observed, as tumours have well-defined margins."
    )

    st.info(
        f"Brain scans which contain the presence of a tumour can be visually "
        f"observed, as tumours have well-defined margins, are oval or "
        f"globular in shape and have variable patterns of enhancement "
        f"regardless of whether they are benign or malignant. These "
        f"properties all have to be translated into machine learning terms "
        f"and these images have to be prepared in a way so that they can be "
        f"put through a model for feature extraction and training."
    )

    st.warning(
        f"As observed from the images found in the visualiser page, the model is "
        f"able to detect differences and to generalise so that it can make "
        f"accurate predictions. A proficient model can train its capacity to "
        f"predict class outcomes on a dataset without adhering too closely to "
        f"that specific set of data. Therefore, the model should be able to "
        f"effectively generalise and predict reliably on any future "
        f"observations. This can be achieved because the model doesn’t just "
        f"memorise the correlation between features and labels in the "
        f"training dataset, but by the general patterns from features to "
        f"labels.\n\n"
        f"However, even though the model showed great results in accuracy "
        f"and in the confusion matrix results there was still some ambiguity "
        f"in the images that were given to the model, as it struggled to "
        f"to make accurate decisions at times."
    )


    st.write("### Hypothesis 2 and Validation")

    st.success(
        f"Comparing the difference between the Sigmoid and SoftMax "
        f"activation functions."
    )

    st.info(
        f"SoftMax and Sigmoid are both activation functions that are used in "
        f"an ML model architecture. Typically, SoftMax is used for "
        f"multi-class classification and Sigmoid is used for binary "
        f"classification. To see which activation function performs best to "
        f"solve our problem, both were tested and compared against each "
        f"other.\n\n"
        f"To obtain a conclusion about which activation function we should "
        f"use in our ML model, a learning curve can be plotted to show the "
        f"accuracy and error rate on the training and validation dataset as "
        f"the model is being trained.\n\n"
        f"For the SoftMax function, it can be observed that there is a lot "
        f"less overfitting as compared to the Sigmoid function."
    )

    st.warning(
        f"In this case, it can be observed that the SoftMax function "
        f"performed better than the Sigmoid function."
    )

    softmax_model = plt.imread(f"assets/streamlit_images/softmax_model.png")
    st.image(softmax_model, caption='SoftMax Loss/Accuracy Performance')

    sigmoid_model = plt.imread(f"assets/streamlit_images/sigmoid_model.png")
    st.image(sigmoid_model, caption='Sigmoid Loss/Accuracy Performance')

