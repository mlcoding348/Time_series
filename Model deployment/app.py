#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 22:00:13 2022

@author: dhanushkobal
"""


import pandas as pd
import numpy as np
import joblib
import streamlit 
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import plotly.graph_objects as go
from timeit import default_timer as timer

url  ="/Users/dhanushkobal/Desktop/2021 Projects/Kaggle/Time series/store-sales-time-series-forecasting/Model deployment project/e_net.pkl"

# loaded the model
model= open(url,"rb")
e_net_model=joblib.load(model)


# uploading the full data
full_data = pd.read_csv('/Users/dhanushkobal/Desktop/2021 Projects/Kaggle/Time series/store-sales-time-series-forecasting/Model deployment project/train.csv')
full_data_1 = full_data[['date' , 'sales']].groupby('date').mean()
full_data_1['type'] = 'exact'

# creating the test set
test = pd.DataFrame({'date' : pd.date_range(start="2017-08-16",end="2017-09-15")})
test['date_of_week_name'] = test['date'].dt.day_name().astype(object)
test['is_weekend'] = np.where(test['date_of_week_name'].isin(['Sunday', 'Saturday']), 1, 0 ).astype(object)
test['id'] = np.arange(2999996.5 + 1*1782, 2999996.5 + (test.shape[0]+1)*1782, 1782)
test = test.drop(['date', 'date_of_week_name'], axis = 1 )    
test = test[['id' ,'is_weekend']]      


# model prediction
def ts_predict(forcast):
    a = int(forcast)
    model_prediction = np.round(e_net_model.predict(test.iloc[:a]),2)
    return model_prediction


def run():
    streamlit.title("TS Model")
    
    html_temp = """
    Hello, in this webpage where you can enter in a number
    and our Time Series model will forecast that much stpes into the future
    """
    
    streamlit.markdown(html_temp)
    var_1 = streamlit.text_input("How many steps you want to forecast into the future?")
    prediction = ""
    
    
    
    if streamlit.button("Predict"):
        prediction = ts_predict(var_1)
        
        #starting timer
        start = timer()
        
        
        # create a dataframe with train + forecast
        start_date = datetime.date(2017, 8 , 16)
        number_of_days = len(prediction)
        date_list = [(start_date + datetime.timedelta(days = day)).isoformat() for day in range(number_of_days)]
        trial = pd.DataFrame({'date':date_list  , 'sales':prediction.tolist(), 'type': 'forecast'})
        trial.set_index('date', inplace=True)
        forecasted_data = pd.concat([full_data_1 , trial], axis = 0)
        forecasted_data['date'] = forecasted_data.index
        forecasted_data.type = forecasted_data['type'].astype('object')
        
        streamlit.dataframe(forecasted_data.drop('date', axis = 1).tail(15))
        
        
        # creating a interactive plot
        # color='type'
        fig = px.line(forecasted_data,
            x = 'date', y= 'sales', title = "Future forecast", color = 'type')
        fig.add_shape(type = 'line',
        x0 = '2017-08-15', x1 = '2017-08-16' , y0 = 427.9809, y1 =  485.1700,
        line = dict(color = 'red'))
        streamlit.plotly_chart(fig)

        #import plotly.graph_objects as go
        #fig = go.Figure()
        #fig = px.scatter(forecasted_data , x =forecasted_data.date, y = forecasted_data.sales, color = 'type')
        #fig.data[0].update(mode='markers+lines')
        
        #streamlit.plotly_chart(fig)
        
        # end timer
        end = timer()
        streamlit.success(f' Total time taken to finish forecasting (ms): {round((end - start), 4)}')
        
    
if __name__ == '__main__':
    run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    