# stock-predictor  
  
  
This week-12 assignment deploys a stock prediction model as a RESTful API using `FastAPI` to an AWS EC2 instance
running `Ubuntu 22.04.1 LTS`, on which miniconda was installed and used to create  a `stock-predictor` conda 
environment using Python 3.8, fastapi uvicorn, pandas, matplotlib, yfinance, pystan, prophet, joblib, and plotly.

This time series model uses `Prophet` to predict stock market prices. A popular forcaster, `Prophet` is 
based on an additive model in which non-linear trends are fit with seasonality & holiday effects. It works
best in the setting of strong seasonal effects; when several seasons of historical data are available; and
is robust to outliers and missing data.

Setup and implementation of `model.py` locally, followed by AWS deployment and ability to make predictions 
interactively via the public IPv4 address of the associated EC2 instance and `FastAPI` are described in detail 
in the accompanying `README_MLE-9.md` file.  

Note that one can trigger and view predictions from the AWS-deployed model via its public IPv4 address from a 
local machine terminal using the `curl` command, as documented in `README_MLE-9.md`, for example: 

```
curl \
>>--header "Content-Type: application/json" \
>>--request POST \
>>--data '{"ticker":"MSFT", "days":7}' \         # in this example, 7 day forecast for MSFT
>>http://52.0.151.180:8000/predict               # public IPv4 address of AWS EC2 instance
```
  

				

