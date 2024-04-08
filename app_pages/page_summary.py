import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"A brain tumour is a growth of cells that occur in or near "
        f"the brain tissue. Brain tutors can develop in any part of "
        f"the brain such as the skull base, brainstem, sinuses and nasal "
        f"cavity. There are over 120 different type of brain tumours and "
        f"they are characterised by their location and they type of cells "
        f"they are made from. Brain tumours can be non-cancerous (benign) "
        f"or cancerous (malignant).\n\n"
        f"**Project Dataset**\n\n"
        f"The available dataset contains a total of 3713 brain MRI scans, in "
        f"which 1789 are brain scans that have a tumour present and 1923 are "
        f"brain scans that have a healthy brain with no tumours present."
    )

    st.write(
        f"For more information, please visit and read the "
        f"[Project README file](https://github.com/lgaudencio/brain-tumor-detection)"
    )

    st.success(
        f"The project has 2 business requirements:\n\n"
        f"1 - The client is interested in conducting a study to visualise the "
        f"difference in MRI scans between a healthy brain and one with a tumour\n\n"
        f"2 - The client is interested in having a reliable and functional ML model "
        f"to predict if an MRI scan of a brain has the absence or presence of a tumour\n\n"
        f"3 - The client is interested in having a dashboard at their disposal to "
        f"obtain a prediction report of the examined MRI brain scans."
    )
