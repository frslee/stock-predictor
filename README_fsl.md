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

___1. How does the Prophet Algorithm differ from an LSTM? Why does an LSTM have poor performance against ARIMA and Prophet for time series?___  

___Ans:___  
  
___a. Prophet___ was developed by Facebook to make predictions on time series values specifically in the business domain. It is a so-called ___additive model___ since it assumes that the value of a time series at any given point is the sum of 4 components viz.:  

```math
\hat{y}_t = g(t) + s(t) + h(t) + \epsilon_t
```
where:  
* $g(t)$ = general trend  
* $s(t)$ = seasonal trend  
* $h(t)$ = holiday trend  
* $\epsilon_t$ = random fluctuations (assumed to follow a normal distribution with μ = 0 and unknown variance $\sigma^2$)
	
	  
___b. ARIMA___ (AutoRegressive Integrated Moving Average) is a second type of time-series model--a subset of linear regression models--that incorporates:  
*  __Past values,__ using the last $p$ time series values as features;  
*  __Past errors,__ using the last $q$ errors made by the model;  
*  __Differences,__ represented by the parameter $d$, which indicates the number of differences $\Delta_{(t, t-1)}, \Delta_{(t-1, t-2)}...\Delta_{(t-d+1, t-d)}$ between consecutive time-series values that must be subtracted from a given time-series value $y_t$ at time $t$ to render it _stationary,_ that is, _independent_ of e.g. seasonal or temporal trends. When this so-called _differencing_ is applied to a time series, the resulting _stationarity_ allows the model to train on an _approximately_ stationary time series, and thus learn _more advanced patterns._
*  _NB:_ The variables, $p, q, d$ are tunable hyperparameters.  
  
    
    
___c. By contrast, LSTMs___ (a more general version of GRUs) are _sequence models_ based on _RNNs,_ in which traditionally, the output $\hat{y}^{(t)}$ at time $t$ depends on the current sequence input, $x^{(t)}$ as well as the activation $a^{(t-1)}$ from the previous time step, which itself results from the combination of all preceding sequential inputs $x^{(0)}...x^{(t-1)}$ together with their corresponding activations $a^{(0)}...a^{(t-2)}$. LSTMs incorporate, in addition, 3 gates (commonly $\Gamma_u$ = update gate, $\Gamma_f$ = forget gate, & $\Gamma_o$ = output gate) at each time step to enable capturing of long-term dependencies in the input sequence.  
  
  

Based on the above, we can make several observations regarding Prophet, ARIMA, & LSTM models specifically for time series (e.g., stock price forecasting):    
*  The recurrent-neural-network-(RNN)-based ___LSTM___ is the most general of the three, and can be applied to modeling any sequence type. As a deep-learning architecture encoding sequence information and long-range dependencies, given sufficient data, time for hyperparameter tuning and training, and computational power, it will generate the best results. If, however, training data is more limited, an LSTM will perform the poorest due to model complexity >> input data → _overfitting._  
*  ___Prophet___, given its simplicity, is easy & fast to implement, requires less hyperparameter tuning (and data pre-processing), and performs well specifically in _business time series_ with a combination of _general trends_ and _temporal patterns on different time scales._  
*  ___ARIMA___--also much simpler than LSTMs--can outperform the latter on smaller datasets but require significant effort and domain expertise to set its hyperparameters $p, q, d$ optimally.
*  
*  
