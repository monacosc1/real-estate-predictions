# Real Estate Price Prediction (SARIMA)

SARIMA time series models forecasting median sale prices across seven U.S. cities, deployed as a Streamlit web application.

## Key Finding

![Multi-City Price Trends](charts/real-estate-prediction-fig1-multi-city-price-trends.png)

*Chicago and Boston offer the strongest predicted ROI based on SARIMA price forecasts across all seven metros.*

## Overview

Predicting where home prices are headed is essential for investment decisions and market timing. This project builds city-specific SARIMA models using Redfin metro-level data, automatically selects optimal model parameters with pmdarima, and deploys forecasts through an interactive Streamlit app where users can select cities and forecast horizons.

## Tools & Technologies

- Python
- Pandas
- statsmodels
- pmdarima
- Streamlit
- Matplotlib

## Results

The models produce 12-month price forecasts for Austin, Boston, Chicago, Denver, Miami, New York, and San Francisco. ROI calculations based on forecast endpoints identify which markets offer the most favorable entry points.

![All Cities Forecast Grid](charts/real-estate-prediction-fig10-all-cities-forecast-grid.png)

*12-month SARIMA forecasts for all seven cities showing predicted median sale price trajectories.*

## Live App

Try the interactive forecast tool: [Real Estate Predictions on Streamlit](https://real-estate-predictions-5zubghkpzgc54tocbxw3dc.streamlit.app/)

## View Full Analysis

For the complete writeup with all charts and methodology, visit the [project page on scottmonaco.com](https://scottmonaco.com/real-estate-prediction).
