

## Dataset Content

The dataset contains a total of 3713 MRI scan images, in which they all consist of a single brain scan and all have a black background as all images are in black and white. These brain scan images are further broken down into 1923 healthy brain scans and 1789 scans of a brain in which a tumour is present. According to [Brain Tumour Research](https://braintumourresearch.org/blogs/campaigning/stark-facts#:~:text=We%20understand%20the%20power%20of%20statistics&text=Too%20many%20people%20are%20being,diagnosed%20with%20a%20brain%20tumour) in the UK, 16,000 people each year are diagnosed with a brain tumour. The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset). 

## Business Requirements 

We have been tasked with developing a machine learning model to be able to detect the presence of brain tumours by using MRI scans. The requested system should be able to detect whether a given scan is healthy or has a tumour present, therefore altering medical staff that the patient needs further treatment. The system was requested by medical professionals to assist them in automating the detection of brain tumours. Medical professionals conduct thousands of brain MRI scans every year all across the country. As a result of this, a manual process is not scalable as more and more scans are being performed. 

The key stakeholders are: 

* Medical and healthcare professionals 
* Healthcare facilities, such as hospitals and clinics 
* Patients 
* Cancer research programs 

To summarise: 

* The client is interested in conducting a study to visualise the difference in MRI scans between a healthy brain and one with a tumour. 
* The client is interested in having a reliable and functional ML model to predict if an MRI scan of a brain has the absence or presence of a tumour. 
* The client is interested in having a dashboard at their disposal to obtain a prediction report of the examined MRI brain scans. 

## The rational to map the business requirements to the Data Visualisations and ML tasks

All three business requirements that were set out in a previous section called “Business Requirements”, have been split into several user stories which were translated into Machine Learning Tasks. 

**Business Requirement 1: Data Visualisation**

“The client is interested in conducting a study to visualise the difference in MRI scans between a healthy brain and one with a tumour.”

User Stories: 
* As a client, I want an interactive dashboard that is easy to navigate so that I can view and understand the data which has been presented. 

* As a client, I want the mean and the standard deviation to be displayed for images that display a healthy scan and a tumour scan, so that I can visually differentiate the MRI scans.

* As a client, I want to display the difference between an MRI scan that is of a healthy brain and an MRI scan of a brain that has the presence of a tumour, so that I can visually differentiate the MRI scans.

* As a client, I want to display an image montage for MRI scans that are healthy and that have the presence of a tumour, so that I can visually differentiate the MRI scans. 

**Business Requirement 2: Classification**

“The client is interested in having a reliable and functional ML model to predict if an MRI scan of a brain has the absence or presence of a tumour.”

User Story: 

* As a client, I want to upload image(s) of the MRI scan results, so that the ML model can give me an immediate and accurate prediction on whether the image is healthy or has a tumour. 

**Business Requirement 3: Reporting**

“The client is interested in having a dashboard at their disposal to obtain a prediction report of the examined MRI brain scans.”

User Story: 

* As a client, I want to obtain a report from the ML predictions on new MRI scans.
