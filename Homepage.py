import os
import streamlit as st

# WİDE
st.set_page_config(layout="wide")

# Function to get a list of prediction files in the City_Predictions folder
def get_prediction_files(folder_path):
    prediction_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_prediction_plot.png"):
            prediction_files.append(os.path.join(folder_path, filename))
    return prediction_files 

# Define the folder where prediction plots are stored
prediction_folder = "city-predictions"
# Get the list of prediction files in the folder
prediction_files = get_prediction_files(prediction_folder)

st.title("Predictions on Next 12 Month")
# Set up the Streamlit app layout in two columns
col1, col2 = st.columns(2)

# Display the prediction plots in both columns (half in each)
with col1:
    for prediction_file in prediction_files[:len(prediction_files)//2]:
        st.image(prediction_file, caption=os.path.basename(prediction_file), use_column_width=True)

with col2:
    
    for prediction_file in prediction_files[len(prediction_files)//2:]:
        st.image(prediction_file, caption=os.path.basename(prediction_file), use_column_width=True)

# Visualize ROI Plot
st.title("Return on Investment")

st.image("city-predictions/roi.png", caption="Return on Investment", use_column_width=True)