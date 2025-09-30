#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


# In[2]:


# Retrieve stock data for Apple Inc. from Yahoo Finance
df=yf.download('TATASTEEL.NS', start='1996-01-01', end='2024-10-18')
data = df['Close']  # Focus on the 'Close' price column


# In[3]:


# Convert the data to a numpy array
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)  # Calculate training data length

# Scale the data to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# In[4]:


# Create training data sets
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # Past 60 days
    y_train.append(train_data[i, 0])      # Current day

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[5]:


# Initialize the LSTM model
model = Sequential()
model.add(LSTM(70, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(70, return_sequences=True))
model.add(LSTM(70, return_sequences=False))
model.add(Dense(45))
model.add(Dense(45))
#model.add(Dense(35))
model.add(Dense(1))  # Final output layer


# In[6]:


# Compile the model with Adam optimizer and MSE loss function
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['r2_score'])


# In[7]:


# Fit the model to the training data
model.fit(x_train, y_train, batch_size=150, epochs=5)


# In[8]:


# Prepare testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])  # Past 60 days

# Convert to numpy array and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[9]:


# Predict the stock prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling


# In[10]:


from sklearn.metrics import r2_score
r2_score(y_test,predictions)


# In[11]:


# Calculate RMSE for model evaluation
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f'Root Mean Squared Error: {rmse}')


# In[12]:


# Visualize the results
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# In[13]:


valid


# In[14]:


# Forecast the next 10 days
last_60_days = scaled_data[-60:]  # Take the last 60 days from the dataset
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))  # Reshape to (1, 60, 1)
# Visualize the forecasted results

forecasted_prices = []

for _ in range(10):  # Forecast for the next 10 days
    predicted_price = model.predict(last_60_days)
    forecasted_prices.append(predicted_price[0, 0])
    
    # Reshape predicted_price to (1, 1, 1) and append it to last_60_days while removing the first element
    predicted_price = np.reshape(predicted_price, (1, 1, 1))  # Make sure it has shape (1, 1, 1)
    last_60_days = np.append(last_60_days[:, 1:, :], predicted_price, axis=1)  # Append along axis=1


# In[15]:


# Inverse scale the forecasted prices back to original scale
forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))


# In[16]:


# Display the forecasted prices for the next 10 days
print(f"Forecasted stock prices for the next 10 days:")
print(forecasted_prices)


# In[17]:


forecasted_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]  # Get next 10 business days
forecast_df = pd.DataFrame(forecasted_prices, index=forecasted_dates, columns=['Forecasted Price'])


# In[18]:


start=df.index[-1] 
apple_quote2=yf.download('TATASTEEL.NS',start,period='1mo',interval='1d')


# In[19]:


actual=apple_quote2['Close']
quote2=pd.DataFrame(actual.iloc[:10,:])

quote3=actual.iloc[:10,:]
# Create a DataFrame for the forecasted prices
forecast_df = pd.DataFrame(forecasted_prices, index=forecasted_dates, columns=['Forecasted Price'])

# Ensure the 'quote2' (actual stock prices) DataFrame only contains the first 10 actual stock prices
quote2 = pd.DataFrame(actual.iloc[:10].values, columns=['Actual Price'], index=forecasted_dates)

# Concatenate forecasted prices and actual prices DataFrames
df_concat = pd.concat([forecast_df, quote2], axis=1)

# Display the concatenated DataFrame
df_concat


# In[20]:


# Calculate RMSE for model evaluation
rmse = np.sqrt(np.mean((forecasted_prices - quote3 )** 2))
print(f'Root Mean Squared Error: {rmse}')


# In[ ]:





# In[ ]:





# In[ ]:




