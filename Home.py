import streamlit as st
#from streamlit_option_menu import option_menu
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#import calendar
#from datetime import date
import torch
import torch.nn as nn
from torch.optim import Adam


# set page config
st.set_page_config(page_title="Stock Price Analyser", page_icon=":moneybag:", layout="wide")

# Define CSS style for navbar
navbar_style = """
    <style>
        .navbar {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            background-color: red;
            padding: 10px;
            margin-bottom: 30px;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            border-radius: 5px;
        }
        .footer {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: left;
            background-color: #FF0000;
            padding: 10px;
            margin-top: 30px;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 16px;
            border-radius: 5px;
        }
        .analyze-button {
            background-color: #FF0000;
            color: #FFFFFF;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
        }
    </style>
"""

# Render navbar
st.markdown(navbar_style, unsafe_allow_html=True)
st.container().markdown('<div class="navbar">Stock Price Analyzer</div>', unsafe_allow_html=True)

st.title("Predict the Stock")

# Add a text input for users to enter the stock symbol
stock_symbol = st.text_input("Enter a stock symbol (e.g. AAPL for Apple):")

start = st.text_input("Enter the starting date in the format YYYY-MM-DD")

end = st.text_input("Enter the ending date in the format YYYY-MM-DD")

# Define a function to get stock data from Yahoo finance
def load_data(symbol):
    data = yf.download(symbol, start, end)
    return data

# Load data for the entered stock symbol
if stock_symbol:
    data = load_data(stock_symbol)

if st.button(" Analyse", key="load_data_button"):
    
    # Load data for the entered stock symbol
    if stock_symbol and start and end:
        data = load_data(stock_symbol)
        if len(data) > 0:
            # Display the loaded data
            st.subheader("Stock Data")
            st.write(data)

            # Calculate the rolling mean and standard deviation of the closing prices
            data["RollingMean"] = data["Close"].rolling(window=20).mean()
            data["RollingStd"] = data["Close"].rolling(window=20).std()

            # Plot the rolling mean and standard deviation
            st.subheader("Rolling Mean and Standard Deviation")
            st.line_chart(data[["Close", "RollingMean", "RollingStd"]])

            # Split the data into training and testing sets
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            #CODE FOR LSTM

            # Prepare the sequential data for LSTM
            train_X = train_data[["Open", "High", "Low", "Close"]].values
            train_y = train_data["Adj Close"].values
            test_X = test_data[["Open", "High", "Low", "Close"]].values
            test_y = test_data["Adj Close"].values

            # Convert the data to PyTorch tensors
            train_X = torch.from_numpy(train_X).float()
            train_y = torch.from_numpy(train_y).float()
            test_X = torch.from_numpy(test_X).float()
            test_y = torch.from_numpy(test_y).float()

            # Reshape the input data to fit LSTM input shape (samples, timesteps, features)
            train_X = train_X.unsqueeze(1)
            test_X = test_X.unsqueeze(1)


            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.lstm = nn.LSTM(input_size, hidden_size)
                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    _, (hidden, _) = self.lstm(x)
                    out = self.fc(hidden[-1])
                    return out

            input_size = train_X.shape[2]
            hidden_size = 64
            output_size = 1
            num_epochs = 10
            learning_rate = 0.001

            model = LSTMModel(input_size, hidden_size, output_size)
            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(train_X)
                loss = criterion(outputs.flatten(), train_y)
                loss.backward()
                optimizer.step()

            # Switch to evaluation mode
            model.eval()

            # Make predictions using the trained LSTM model
            with torch.no_grad():
                test_predictions = model(test_X)

            # Create a new DataFrame for predictions
            predictions_df = pd.DataFrame({"Predictions LSTM": test_predictions.flatten().detach().numpy()}, index=test_data.index)

            # Concatenate predictions with the test_data DataFrame
            test_data = pd.concat([test_data, predictions_df], axis=1)
            


            if len(train_data) > 0 and len(test_data) > 0:
                    # Train a linear regression model using the training data
                    model = LinearRegression()
                    model.fit(train_data[["Open", "High", "Low", "Close"]], train_data["Adj Close"])

                    # Make predictions using the testing data
                    test_data["Predictions"] = model.predict(test_data[["Open", "High", "Low", "Close"]])

                    # Calculate the mean squared error of the predictions
                    mse = np.mean((test_data["Predictions"] - test_data["Adj Close"]) ** 2)

                    # Display the predicted and actual closing prices
                    st.subheader("Predictions vs Actual")
                    st.write(test_data[["Adj Close", "Predictions"]])

                    # Display the mean squared error
                    st.subheader("Error Accuracy")
                    st.write(mse)

                    # Display a buy or sell recommendation based on the predicted prices
                    if test_data["Predictions"].iloc[-1] > test_data["Adj Close"].iloc[-1]:
                        st.subheader("Recommendation:")
                        st.write(f'<span style="color:red">Based on our model, we recommend sell investment</span>', unsafe_allow_html=True)
                    else:
                        st.subheader("Recommendation:")
                        st.write(f'<span style="color:green">Based on our model, we recommend to buy</span>', unsafe_allow_html=True) 
            else:
                st.write("No data available for the entered stock symbol and date range.")
        else:
            st.write("Please fill in all required fields.") 

# Render footer
st.markdown(navbar_style, unsafe_allow_html=True)
st.container().markdown('<div class="footer">© 2023 - Stock Price Analyzer</div>', unsafe_allow_html=True)