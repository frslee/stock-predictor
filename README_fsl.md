# ___Week 12: Stock-predictor___  

```
AWS EC2 instance name:    fslMLE-9
public IPv4:              52.0.151.180
Inbound rules:            (as per instructions)
OS:                       Ubuntu 22.04.1 LTS (x64)
kernel:                   5.15.0.-1022.aws
installations:            Git
                          miniconda3 → conda 'stock-predictor' env (Python 3.8)
```

## A. Deliverables included:  
*  `model.py` and `main.py` scripts  
*  `requirements.txt` (as per instructions)  
*  This `README_fsl.md` file, which supplements the original course README (renamed to `README_MLE9.md` in repo) with:  
    *  Further implementation details
    *  Description of how to `cURL` API endpoint  
    *  Answers to Rubric questions  


## B. Overview:  
  
This week-12 assignment deploys a stock prediction model as a RESTful API using `FastAPI` to an AWS EC2 instance running `Ubuntu 22.04.1 LTS`, on which miniconda was installed and used to create  a `stock-predictor` conda environment using Python 3.8, fastapi uvicorn, pandas, matplotlib, yfinance, pystan, prophet, joblib, and plotly.

This time series model uses `Prophet` to predict stock market prices. A popular forcaster, `Prophet` is based on an additive model in which non-linear trends are fit with seasonality & holiday effects. It works best in the setting of strong seasonal effects; when several seasons of historical data are available; and is robust to outliers and missing data.

Setup and implementation of `model.py` locally, followed by AWS deployment and ability to make predictions  interactively via the public IPv4 address of the associated EC2 instance and `FastAPI` are described in detail  in the original MLE-9 course repo `README_MLE9.md` file for this week's assignment.  

Note that one can trigger and view predictions from the AWS-deployed model via its public IPv4 address from a local machine terminal using the `curl` command, as documented in `README_MLE9.md`; for example: 

```
curl \
>>--header "Content-Type: application/json" \
>>--request POST \
>>--data '{"ticker":"MSFT", "days":7}' \    # in this example, 7 day forecast for MSFT
>>http://52.0.151.180:8000/predict          # public IPv4 address of AWS EC2 instance
```
  

## C. Answers to Rubric Questions:  

___1. How does the Prophet Algorithm differ from an LSTM? Why does an LSTM have poor performance against ARIMA and Profit for time series?___  

___Ans:___  
`Prophet` was developed by Facebook to make predictions on time series values specifically in the business domain. It is a so-called ___additive model___ since it assumes that the value of a time series at any given point is the sum of 4 components viz.:  

```math
\hat{y}_t = g(t) + s(t) + h(t) + \epsilon_t
```
where:  
* $g(t)$ = general trend  
* $s(t)$ = seasonal trend  
* $h(t)$ = holiday trend  
* $\epsilon_t$ = random fluctuations (assumed to follow a normal distribution with μ = 0 and unknown variance $\sigma^2$)

_By contrast,_ ___LSTMs___ are _sequence models_ based on _RNNs,_ in which traditionally, the output $\hat{y}^{(t)}$ at time $t$ depends on the current sequence input, $x^{(t)}$ as well as the activation $a^{(t-1)}$ resulting from all preceding sequential inputs $x^{(0)}...x^{(t-1)}$. LSTMs incorporate, in addition, 3 gates at each time point that enable capturing long-term dependencies in the input sequence.  
	


