import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_mri_scan_visualizer_body():
    st.write("### MRI Scan Visualizer")
    st.info(
        f"The client is interested in conducting a study to visualise the "
        f"difference in MRI scans between a healthy brain and one with a tumour."
    )

    version = 'v1'
    if st.checkbox("Difference between average and variability image"):

        avg_tumor = plt.imread(f"outputs/{version}/avg_var_tumor.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            f"We notice the average and variability images did show some "
            f"general patterns where we could intuitively differentiate one "
            f"from another. However, there are instances where the difference "
            f"is not very obvious, this can be due to size and axis in which "
            f"the scan was taken. Therefore, these factors can make the "
            f"prediction quite challenging."
        )

        st.image(avg_tumor, caption='Tumor MRI Scan - Average and Variability')
        st.image(avg_healthy, caption='Healthy MRI Scan - Average and Variability')
        st.write("---")

    if st.checkbox("Differences between average tumor and average healthy MRI brain scans"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"We notice this study didn't show patterns where "
            f"we could intuitively differentiate one from another.")
        st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
      st.write("To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/mriscans_dataset/mri-scans'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")
