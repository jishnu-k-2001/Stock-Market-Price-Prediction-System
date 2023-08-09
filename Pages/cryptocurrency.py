import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Add navigation bar
# Define CSS style for navbar
navbar_style = """
    <style>
        .navbar {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            background-color: #FF0000;
            padding: 10px;
            margin-bottom: 30px;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            border-radius: 5px;
        }
    </style>
"""

# Render navbar
st.markdown(navbar_style, unsafe_allow_html=True)
st.container().markdown('<div class="navbar">Stock Price Analyzer</div>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Cryptocurrency to analyze
crypto_name = st.sidebar.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD'])




# Define CSS style for select box
#select_box_style = """
#    <style>
        #crypto_select:focus {
#            color: #FF0000 !important;
#        }
#    </style>
#"""

# Render select box with modified style
#st.sidebar.markdown(select_box_style, unsafe_allow_html=True)




# Start and End dates for analysis
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime('2017-01-01'))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime('2022-04-01'))

# Retrieve data from Yahoo Finance API
crypto_data = yf.download(crypto_name, start=start_date, end=end_date)

# Data preprocessing
crypto_data.reset_index(inplace=True)
crypto_data['Timestamp'] = crypto_data['Date'].apply(lambda x: pd.Timestamp(x).timestamp())
X = np.array(crypto_data['Timestamp']).reshape(-1, 1)
y = np.array(crypto_data['Close'])

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction_date = pd.date_range(start=end_date, periods=30, freq='D')
prediction_date = prediction_date.map(lambda x: pd.Timestamp(x).timestamp())
prediction = model.predict(prediction_date.values.reshape(-1, 1))

# Graph of historical and predicted prices
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(crypto_data['Timestamp'], crypto_data['Close'], label='Historical Prices')
ax.plot(prediction_date, prediction, label='Predicted Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title('Historical and Predicted Prices of {}'.format(crypto_name))
ax.legend()
st.pyplot(fig)

# Recommendations
st.header('Recommendations')
if prediction[-1] > y[-1]:
    st.write('Based on our model, we recommend holding on to your investment in {} for now.'.format(crypto_name))
else:
    st.write('Based on our model, we recommend selling your investment in {} at this time.'.format(crypto_name))
     