#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
from keras.optimizers import Adam


# In[52]:


# Retrieve stock data for Apple Inc. from Yahoo Finance
df=yf.download('TSLA', start='2011-01-01', end='2024-10-20')
data = df['Close']  # Focus on the 'Close' price column


# In[53]:


# Convert the data to a numpy array
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.7)  # Calculate training data length

# Scale the data to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# In[54]:


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


# In[55]:


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Prevent overfitting
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))  # Prevent overfitting
model.add(Dense(50))  # Dense layer to capture trends
model.add(Dense(1))  # Output layer
optimizer = Adam(learning_rate=0.001)  # Tune learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')


# In[56]:


# Compile the model with Adam optimizer and MSE loss function
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['r2_score'])


# In[75]:


# Fit the model to the training data
model.fit(x_train, y_train, batch_size=32, epochs=5)


# In[76]:


# Prepare testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])  # Past 60 days

# Convert to numpy array and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[77]:


# Predict the stock prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling
predictions.shape


# In[78]:


from sklearn.metrics import r2_score
r2_score(y_test,predictions)


# In[79]:


# Calculate RMSE for model evaluation
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f'Root Mean Squared Error: {rmse}')


# In[80]:


# Visualize the results
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# In[81]:


valid


# In[82]:


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


# In[83]:


# Inverse scale the forecasted prices back to original scale
forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))


# In[84]:


# Display the forecasted prices for the next 10 days
print(f"Forecasted stock prices for the next 10 days:")
print(forecasted_prices)


# In[85]:


forecasted_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]  # Get next 10 business days
forecast_df = pd.DataFrame(forecasted_prices, index=forecasted_dates, columns=['Forecasted Price'])


# In[86]:


#import yfinance as yf
#apple_quote=yf.download('AAPL',start='2012-01-01',end='2019-12-17')
#new_df=apple_quote['Close']
#last_60_days=new_df[-60:].values
#last_60_days_scaled=scaler.transform(last_60_days)
#X_test=[]
#X_test.append(last_60_days_scaled)
#X_test=np.array(X_test)
#X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#pred_price=model.predict(X_test)
#pred_price=scaler.inverse_transform(pred_price)
#print(forecasted_prices)


# In[87]:


start=df.index[-1] 
apple_quote2=yf.download('TSLA',start,period='1mo',interval='1d')


# In[88]:


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


# In[89]:


# Calculate RMSE for model evaluation
rmse = np.sqrt(np.mean((forecasted_prices - quote3 )** 2))
print(f'Root Mean Squared Error: {rmse}')







