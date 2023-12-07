# Real-Estate-Prediction

Predictions on Median Real Estate Prices on US Cities 
[Click here to access the Real Estate App](https://real-estate-predictions-5zubghkpzgc54tocbxw3dc.streamlit.app/)
General Data Source: [Redfin Data Center](https://www.redfin.com/news/data-center/)

**Metro Level Data Source**: [redfin_metro_market_tracker.tsv000](https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/state_market_tracker.tsv000.gz)

Metro Level Data: Utilizing the Redfin Metro Market Tracker Data, we focus on metro-level granularity. This allows for a detailed analysis, crucial for understanding regional real estate dynamics.

## Dashboard
* **Created Using** Streamlit
* **Link to Dashboard** https://real-estate-predictions-5zubghkpzgc54tocbxw3dc.streamlit.app/

![alt text](https://github.com/monacosc1/real-estate-predictions/blob/master/images/dashboard_screenshot.png) 

## Notebook Descriptions 

### Data Preparation for Prediction.ipynb
This script provides a comprehensive analysis of real estate market data, focusing on achieving stationarity, a crucial prerequisite for time series analysis. It begins by loading and preprocessing the data, extracting relevant columns, and calculating data availability statistics. The script then delves into stationarity analysis, explaining its importance and methods to check for stationarity, including differencing. Key visualizations, such as ACF and PACF plots, alongside the ADF statistic and P-value comparison, are presented to aid in understanding data processing steps and their impact on stationarity. This documentation serves as a valuable resource for analyzing time series data in the context of real estate market trends.

### SARIMA Model.ipynb

## Data Loading and Preprocessing:
Reads data from a CSV file into a Pandas DataFrame.
Defines a function to prepare time series data for modeling, including differencing and splitting into training and testing sets.

## Time Series Modeling:
Utilizes SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling to forecast median sale prices for various cities.
Applies automated SARIMA model selection using pmdarima.auto_arima.
Defines functions to perform SARIMA training and prediction.

##  Visualization and Plotting:
Creates a function to plot actual vs. predicted values for median sale prices in different cities.
Generates prediction plots for specified cities and saves them to a folder.

##  ROI (Return on Investment) Computation and Visualization:
##Computes the ROI for each city based on the last observed median sale price and the predicted values.
Visualizes cities with positive and negative ROI separately in bar plots.

# Steps to Reproduce Results:

- Download the data from provided links and put the datas same folder with notebooks.
- Run the notebooks in order.
- After generating the image files using the provided Jupyter Notebook, make sure to move these  files into the same folder structure provided in the GitHub repository. This ensures that the figures can be properly accessed and displayed when users clone or download the repository.


### Initializing the Web App for Real Estate Market Analysis

To view the generated app by this project, follow these steps:


1. Clone the repository

    ```bash
    git clone https://github.com/enesbol/Real-Estate-Prediction
    ```
 
2. Navigate to the repository's root directory using your terminal. 
 
3. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    pip install pmdarima
    ```

2. Run the Streamlit Web App:

    ```bash
    streamlit run Homepage.py
    ```

3. This will launch the Streamlit web app and display the predictions in your default web browser.

First Page:
Precomputed 12-Month Predictions: Visualizations of precomputed 12-month predictions for each city, offering a quick overview of future trends in the real estate market.
Overall ROI (Return on Investment) Plot:A plot showcasing the Return on Investment (ROI) for different cities.

Second Page:
City-Specific Predictions: Ability to make predictions for a chosen city and set the desired number of months for forecasting.
Interactive Prediction Graphs: Interactive graphs displaying predicted median sale prices, providing a visual representation of the predicted market trajectory.
ROI Calculation: Calculation and display of Return on Investment (ROI) based on the predicted median sale prices, assisting users in investment decision-making.
Download Predictions: A download button to obtain the predictions, allowing users to save the forecasted data for further analysis or reference.





