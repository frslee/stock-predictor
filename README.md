# ___Week 13: Build Container Image for FastAPI Stock Prophet___  
  
```
Docker version:     20.10.20 bld 9fdeb9c (installed on Windows)
Host OS:            Windows 10 x64, WSL 2 with Ubuntu 22.04
.dockerignore:      file not required since .pem key pair stored outside
                        project directory root (per Sina)
conda env:          'stock-predictor' (Python 3.8, as per MLE-9 README.md)
```

## A. Overview  
### _i. A `Dockerfile` was created in project root, specifying:_
* `Python 3.8`
* `python3-dev, apt-utils, python-dev, build-essential`
* `pip, cython, numpy, pystan`
* `prophet`
* `fastapi, joblib, pandas, uvicorn, yfinance`
* contents of `src/ .`
    * `main.py`
    * `model.py`
    * `AAPL.joblib, GOOG.joblib, MSFT.joblib`
* `CMD` to run `uvicorn` server  
  
### _ii. The above `Dockerfile` was used to build a __Docker image__ via:_  
```
docker build -t stock-prophet
```  

### _iii. A __Docker container__ was then started using the above Docker image via:_  
```
docker run -d --rm --name stock-prophet-container -p 8000:8000 stock-prophet
```

### _iv. Finally, the stock-predictor app was used to __get predictions__ e.g.,:_  
```
curl --header "Content-Type: application/json --request POST \
     --data '{"ticker":"MSFT","days":7}' http://localhost:8000/predict  
```  

## B. Answers to Rubric Questions:  
---  

___1. What does it mean to create a Docker image and why do we use Docker images?___  
___Ans:___
* When creating a __Docker image__--in this week's exercise, using a plain-text __Dockerfile__--we assemble together the collection of files (installations, application code, dependencies, etc.) that are required to configure a fully operational __container__ environment (see below). A Dockerfile specifies the steps required to create and run a Docker image, with each of its instructions defining a __layer in a hierarchy of layers__ in the image. In turn, a __Docker image__ _(read only),_ when run, becomes an _instance_ of a __container__, which is _live, executable content that users can interact with._
  
* We use __Docker images__ as read-only templates that define __containers,__ which conveniently contain everything an application needs to run. As such, they can be deployed and instantiated quickly, relatively inexpensively (notably as a fast, lightweight, and cost-effective alternative to hypervisor-based virtual machines), at any scale, and most importantly, ___reliably in any environment with confidence that the application code will run.___    

___2. What is the difference between a Container and a Virtual Machine?___  
___Ans:___   

* As mentioned above, __containers__ are fast, lightweight, and cost-effective alternatives to hypervisor-based virtual machines (VMs). In contrast to VMs, which each contain full _guest & host OS's_ running on virtual hardware created and allocated by a __hypervisor__ _(Type I)_ (so-called ___hardware-level virtualization,___ enabling _isolation of VMs_), __containers__ represent ___os-level virtualization___ and permit _isolation of processes._ In other words, instead of abstracting hardware, containers abstract the OS.   
  
* Docker containers run on a single physical server, _sharing the host OS amongst them._  
  
* Because Docker containers are _self-contained,_ and package up an application together with all necessary libraries, dependencies, etc., they can be easily deployed as a single entity and are highly hardware-compatible and readily maintained--in short they exhibit __highly portability.__  
  
* _NB: Since Docker containers are lightweight and don't virtualize hardware, one can run a docker container in a VM!_  
  
___3. What are 5 examples of container orchestration tools?___    
___Ans:___  
* __Orchestrators__ are tools used to manage, scale, and maintain containerized applications. Five examples are:  
    *  __Kubernetes:__ Enables horizontal scaling of a web app from e.g. a single server to several 'nodes,' enabling for example, _load balancing._    
      
    *  __Docker Swarm:__ Converts multiple docker instances into a single virtual host, enabling orchestrating clusters of Docker engines.  
      
    *  __Amazon Elastic Container Service (ECS):__ An AWS container service enabling secure, highly scalable Docker container orchestration.  
      
    *  __Amazon EKS:__ Provides a fully managed orchestration tool for Kubernets clusters on AWS.  
      
    *  __Azure Kubernetes Service:__ Similar to Amazon EKS, AKS provides a fully managed Kubernetes service on Microsoft Azure.  
      
___4. How does a Docker image differ from a Docker container?___  
___Ans:___  
In short, a __Docker image__ _(read only),_ when run, becomes an _instance_ of a __Docker container__, which is _live, executable content that users can interact with (see above answer to §Question 1 for further details)._  
  

---  
  



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
---  


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
*  ___Prophet___, given its simplicity, is easy & fast to implement, requires less hyperparameter tuning (and data pre-processing), and performs well specifically in _business time series_ which exhibit a combination of _general, overall trend(s)_ and _temporal patterns on different time scales._  
*  ___ARIMA___—also much simpler than LSTMs—can outperform the latter on smaller datasets, but require significant effort and domain expertise to set its hyperparameters $p, q, d$ optimally.  

  
  
___2. What is exponential smoothing and why is it used in Time Series Forecasting?___  
___Ans:___  
  
While ARIMA family models make predictions based on a weighted linear sum of recent past observations, ___exponential smoothing___ is a type of univariate time-series forecasting that uses a weighted sum of past observations in which the ___weights decrease exponentially.___ This approach results in predictions that are based more heavily on recent data values. Thus, exponential smoothing performs well in making shorter-term forecasts, specifically in scenarios where the time-scale of changes in a series' parameters exceeds that of the targeted forecast period.  
  
    
 ___3. What is stationarity? What is seasonality? Why is stationarity important in time series forecasting?___  
 ___Ans:___ _Please see above answer to Question §1b._  
   
     
       
 ___4. How is seasonality different from cyclicality? Fill in the blanks:  '.....' is predictable, whereas '.....' is not.___  
 ___Ans:___  
   
 *  ___Seasonality___ _(in time series)_ refers to a ___predictable___ change that occurs regularly based on a particular season, generally within a one-year period. _Example: Retail sales often exhibit seasonality in being highest during Q4 of the calendar year._  
 *  ___Cyclicality___ _(in time series), by contrast,_ refers to an ___unpredictable change that exhibits irregular periodicity and variable duration over time.___ Cyclical phenomena can be shorter in duration than a calendar year, but are often much longer.   
 *  _NB: Seasonality and cyclicality can occur together, for example, when an aperiodic, larger trend varying over years (and of indeterminate duration), also exhibits regular, superimposed smaller bumps (or dips) that are seasonal (predictable) within any given year of the larger, cyclical trend._  
 *  ___In sum, seasonality is predictable, whereas cyclicality is not!___
