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


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # Subset the class you are interested to display
  if label_to_display in labels:

    # Checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # Create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # Create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(
        f"Width {img_shape[1]}px x Height {img_shape[0]}px"
      )
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")