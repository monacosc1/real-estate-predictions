import plotly.graph_objects as go
from matplotlib.ticker import ScalarFormatter
from dateutil.relativedelta import relativedelta
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import streamlit as st
import os

# Disable warnings
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide") # , initial_sidebar_state="collapsed"

def prepare_data(city_df, column_name, test_size, diff_order=1):
    """
    Prepare time series data for modeling.
    
    Args:
        city_df (pd.DataFrame): DataFrame containing time series data for a city.
        column_name (str): Name of the column to model.
        test_size (float): Proportion of data to use for testing.
        diff_order (int): Order of differencing (default is 1 for first-order differencing).
   
    Returns:
        tuple: A tuple containing training and testing datasets.
    """
    # Perform differencing
    city_df[column_name] = city_df[column_name].diff(diff_order)
    city_df.dropna(inplace=True)
    
    # Split data into training and testing sets
    train_size = int(len(city_df) * (1 - test_size))
    train_data, test_data = city_df[:train_size], city_df[train_size:]
    
    return train_data, test_data

def compute_roi(df,converted_predictions):
    last_price_city =  df.groupby('city').last()['median_sale_price'].to_frame().reset_index()
    last_price_predicted = converted_predictions.groupby('City').last()['Predicted_Value'].to_frame().reset_index()

    last_price_city = last_price_city.sort_values('city')
    last_price_predicted = last_price_predicted.sort_values('City')

    # rename  last_price_predicted City as city
    last_price_predicted = last_price_predicted.rename(columns={'City':'city'})

    # merge on city
    last_price = pd.merge(last_price_city, last_price_predicted, on='city', how='inner')

    last_price['ROI'] = ((last_price['Predicted_Value'] - last_price['median_sale_price']) / last_price['median_sale_price']) * 100
    
    return last_price



# Define SARIMA parameters for each city
sarima_parameters = {
    'Austin': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (4, 0, 1)
    },
    'Boston': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (4, 0, 2)
    },
    'Chicago': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (4, 0, 1)
    },
    'Denver': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (0, 0, 1)
    },
    'Miami': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (0, 0, 0)
    },
    'New York': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (0, 0, 3)
    },
    'San Francisco': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (3, 0, 3)
    },
    'Average of 7 City': {
        'seasonal_order': (1, 1, 1, 12),
        'order': (0, 0, 2)
    }
}


def sarima_train_predict(train_data, seasonal_order, forecast_steps, city_name_predict):
    """
    Apply an automated SARIMA (Seasonal Autoregressive Integrated Moving Average) model to time series data and make forecasts.

    Args:
        train_data (pd.Series): Time series training data.
        seasonal_order (tuple): Seasonal order for SARIMA (default is no seasonality).
        forecast_steps (int): Number of steps to forecast into the future. (default is 12 months)

    Returns:
        pd.Series: A Series containing the forecasted values with confidence intervals.
    """
    # Automatically select SARIMA model using pmdarima
    #model = pm.auto_arima(train_data, seasonal=True, m=12, stepwise=True,
    #   suppress_warnings=True, error_action="ignore")

    # Extract seasonal order (P, D, Q, s) from the model
    #P, D, Q, s = model.seasonal_order

    # Extract non-seasonal order (p, d, q) from the model
    #p, d, q = model.order

    # Print
    #print('Seasonal Order:', seasonal_order)
    #print('Order:', (p, d, q))
    
    P, D, Q, s = sarima_parameters[city_name_predict]['seasonal_order']
    p, d, q = sarima_parameters[city_name_predict]['order']

    # Create a SARIMA model (SARIMAX is a more generalized class) 
    sarima_model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    
    # Fit the SARIMA model to the training data
    sarima_results = sarima_model.fit(disp=False)
    
    # Make forecasts
    forecast_series = sarima_results.get_forecast(steps=forecast_steps).summary_frame()
    
    return forecast_series


def convert_to_original_scale(city_predictions, main_data, city):
    # Create a DataFrame to store the converted predictions
    
    converted_predictions = pd.DataFrame(columns=['City', 'Date', 'Predicted_Value', 'Lower_CI', 'Upper_CI'])
    city_data = main_data[main_data['city'] == city]
    city_data['period_end'] = pd.to_datetime(city_data['period_end'])
    first_predicted_date = city_predictions['Date'].loc[0]
    
    last_date_before_prediction = first_predicted_date - pd.DateOffset(months=1)

    # Filter city_data to get the last value before prediction based on year and month
    last_value_before_prediction = city_data[(city_data['period_end'].dt.year == last_date_before_prediction.year) & (city_data['period_end'].dt.month == last_date_before_prediction.month)]['median_sale_price'].iloc[0]

    last_date_before_prediction = city_data[(city_data['period_end'].dt.year == last_date_before_prediction.year) & (city_data['period_end'].dt.month == last_date_before_prediction.month)]['period_end'].iloc[0]

    city_predictions_city = city_predictions[city_predictions['City'] == city].copy()
    
    # Inverse differencing for Predicted_Value
    city_predictions_city['Predicted_Value'] = city_predictions_city['Predicted_Value'].cumsum() + last_value_before_prediction
    
    # Inverse differencing for Lower_CI
    city_predictions_city['Lower_CI'] = city_predictions_city['Lower_CI'].cumsum() + last_value_before_prediction
    
    # Inverse differencing for Upper_CI
    city_predictions_city['Upper_CI'] = city_predictions_city['Upper_CI'].cumsum() + last_value_before_prediction
    
    converted_predictions = pd.concat([converted_predictions, city_predictions_city], ignore_index=True)

    return converted_predictions

def visualize_positive_roi_plotly(last_price):
    # Sort the data by ROI in ascending order
    sorted_data = last_price.sort_values(by='ROI', ascending=False)

    # Create a bar chart
    fig = go.Figure(go.Bar(
        x=sorted_data['city'],
        y=sorted_data['ROI'],
        marker_color='lightblue',
        width=0.5  # Adjust the bar width as needed
    ))

    fig.update_layout(
        title={
            'text': 'ROI',
            'y': 0.95,
            'x': 0.52,
            'xanchor': 'center',
            'yanchor': 'middle',
            'font_size': 24  # Adjust the font size of the title
        },
        xaxis_title='City',
        yaxis_title='ROI (%)',
        xaxis=dict(
            tickvals=list(range(len(sorted_data))),
            ticktext=sorted_data['city'],
        #    tickangle=45,
            tickfont=dict(size=20)  # Adjust the font size of xticks
        ),
        margin=dict(l=50, r=50, b=50, t=50),  # Adjust margins as needed
        height=600,  # Adjust the height of the plot
        width=800,   # Adjust the width of the plot
        template='plotly_white'
    )

     
    # Add ROI values on top of the bars
    for i, roi in enumerate(sorted_data['ROI']):
        
        if roi < 0:
            roi_tick_y_position = roi -1 
        else:
            roi_tick_y_position = roi + 1
        
        fig.add_annotation(
            x=i,
            y=roi_tick_y_position,
            text=f'{roi:.2f}%',
            showarrow=False,
            font=dict(size=20),  # Adjust the font size of the value on top of the bar
            xanchor='center',
            yanchor='bottom',
            yshift=10  # Adjust the vertical position of the value on top of the bar
        )

    return fig


def plot_actual_vs_predicted_plotly(df, predictions_df, city_name, column_name, save_path=None):
    # Filter data for the specified city and column
    city_data = df[(df['city'] == city_name) & (df['property_type'] == 'All Residential')]
    
    # Extract actual values and dates
    actual_values = city_data[column_name]
    actual_dates = city_data['period_end']
    
    # Extract predicted values, lower and upper confidence intervals, and dates
    predictions = predictions_df[(predictions_df['City'] == city_name)]['Predicted_Value']
    lower_ci = predictions_df[(predictions_df['City'] == city_name)]['Lower_CI']
    upper_ci = predictions_df[(predictions_df['City'] == city_name)]['Upper_CI']
    prediction_dates = predictions_df[(predictions_df['City'] == city_name)]['Date']
    
    # Create the figure
    fig = go.Figure()
    
    # Plot the actual values
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_values, mode='lines+markers', name='Actual'))
    
    # Plot the predicted values with confidence intervals
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines+markers', name='Predicted'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=lower_ci, fill=None, mode='lines', line_color='rgba(255,165,0,0.4)', name='Lower CI'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=upper_ci, fill='tonexty', mode='lines', line_color='rgba(255,165,0,0.4)', name='Upper CI'))
    
    num_of_prediction_months = predictions.shape[0]
    fig.update_layout(
    title={
        'text': f'{city_name} - Median Sale Prediction for next {num_of_prediction_months} Months',
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font_size=24,  # Adjust the font size as needed
    xaxis_title='Year',
    yaxis_title=column_name,
    legend=dict(x=0, y= 1, traceorder='normal', orientation='h'),
    height=600,  # Adjust the height of the plot
    width=800,   # Adjust the width of the plot
    margin=dict(t=20),
    xaxis=dict(showline=True, showgrid=False),
    yaxis=dict(showline=True, showgrid=True, gridcolor='lightgray'),
    template='plotly_white'
    )

    # Save the plot if save_path is provided
    if save_path:
        fig.write_image(save_path)
    
    # Show the plot
    return fig


data_path = "./data/data_for_prediction.csv"
df = pd.read_csv(data_path)
df['period_end'] = pd.to_datetime(df['period_end'])


# Sidebar for user input
st.sidebar.title("Select City")
city = st.sidebar.selectbox("Select a City", df['city'].unique())
column_name = 'median_sale_price'  # Adjust this to the column you want to predict
forecast_steps = st.sidebar.slider("Number of Months to Forecast", min_value=1, max_value=24, value=12)


def make_predictions(df, city, forecast_steps):
    
    # Prepare data
    city_df = df[df['city'] == city].copy()
    column_name = 'median_sale_price'  # Adjust this to the column you want to predict
    train_data, test_data = prepare_data(city_df, 'median_sale_price', test_size=0, diff_order=1) # 2.order difference can be tested. 1. order is enough for most cases.
    train_data.reset_index(drop=True, inplace=True)
    
    last_date = train_data['period_end'].max()
    train_data = train_data[column_name]

    #st.write(train_data)
    # Train and make predictions
    forecast_series = sarima_train_predict(train_data, seasonal_order=(1, 1, 1, 12), forecast_steps=forecast_steps, city_name_predict=city)

    
    # Calculate confidence intervals
    forecast_mean = forecast_series['mean']
    forecast_ci = forecast_series[['mean_ci_lower', 'mean_ci_upper']]

    
    from pandas.tseries.offsets import DateOffset

    predicted_date_range = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'City': [city] * forecast_steps,
        'Date': predicted_date_range,
        'Predicted_Value': forecast_mean.values,
        'Lower_CI': forecast_ci['mean_ci_lower'].values,
        'Upper_CI': forecast_ci['mean_ci_upper'].values
    })
    
    return predictions_df

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


if  city and forecast_steps:
    predictions_df = make_predictions(df, city, forecast_steps)
    #st.write(predictions_df,"asdasd")
    if predictions_df.shape[0] > 0:
        
        # Convert predictions to  original scale
        converted_predictions = convert_to_original_scale(predictions_df, df, city)
        last_price = compute_roi(df, converted_predictions)

        prediction_plot_plotly = plot_actual_vs_predicted_plotly(df, converted_predictions, city, column_name)

        roi_fig_plotly = visualize_positive_roi_plotly(last_price)

        col1, col2 = st.columns([4, 1])
        with col1:   
            # Show Chart
            st.plotly_chart(prediction_plot_plotly,use_container_width=True)
        
        with col2:
            # Show ROI Chart
            st.plotly_chart(roi_fig_plotly,use_container_width=True)
        
        
        # Display predictions
        st.title(f"Predictions for {city}")
        st.dataframe(converted_predictions)

        converted_predictions_download = convert_df(converted_predictions)

        st.download_button(
        label=f"Download Prediction data as CSV",
        data= converted_predictions_download ,
        file_name=f'Predictions for {city}.csv',
        mime='text/csv')
