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
        f"As it can be observed from the above images, the model is able "
        f"to detect differences and to generalise so that it can make "
        f"accurate predictions. A proficient model can train its capacity to "
        f"predict class outcomes on a dataset without adhering too closely to "
        f"that specific set of data. Therefore, the model is able to "
        f"effectively generalise and predict reliably on any future "
        f"observations. This can be achieved because the model didn't just "
        f"memorise the correlation between features and labels in the "
        f"training dataset, but by the general patterns from features to "
        f"labels."
    )