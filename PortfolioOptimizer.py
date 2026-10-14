# -*- coding: utf-8 -*-
"""Portfolio Optimizer.

Finds stock portfolio weights with the best Sortino ratio (annualized return
divided by annualized downside deviation) using a vectorized Monte Carlo
search, then runs a year-by-year walk-forward backtest against an
equal-weight portfolio and individual stocks.

Originally written in Google Colab:
    https://colab.research.google.com/drive/1nNhMP-X8ZaLGWz5jq-eJvxpLlc-Bj2zn

Price data comes from Yahoo Finance via ``yfinance``. All downloads go through
``download_prices``, which retries failed requests with exponential backoff,
validates that every ticker returned data, and logs what happened.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import scipy.stats as stats
import logging
import time
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("portfolio_optimizer")


def download_prices(tickers, start, end, max_retries=3, backoff_seconds=2.0):
  """Download daily price data from Yahoo Finance with retries and validation.

  Args:
    tickers: Space-separated ticker string (e.g. "cost aapl nvda") or a list
      of ticker symbols.
    start: Start date as "YYYY-MM-DD" (inclusive).
    end: End date as "YYYY-MM-DD" (exclusive).
    max_retries: How many times to attempt the download before giving up.
    backoff_seconds: Base wait time between attempts. The wait doubles after
      each failure (2s, 4s, 8s, ...) to stay under Yahoo's rate limits.

  Returns:
    A pandas DataFrame with a "Date" column and multi-index price columns
    such as ("Close", "AAPL").

  Raises:
    RuntimeError: If every attempt fails or returns no data.
    ValueError: If any requested ticker comes back with no price data
      (for example, a misspelled or delisted symbol).
  """
  if isinstance(tickers, str):
    tickerList = tickers.upper().split()
  else:
    tickerList = [t.upper() for t in tickers]

  data = None
  for attempt in range(1, max_retries + 1):
    try:
      logger.info("Downloading %s from %s to %s (attempt %d/%d)",
                  ", ".join(tickerList), start, end, attempt, max_retries)
      data = yf.download(tickers, start, end, progress=False)
    except Exception as exc:
      logger.warning("Yahoo Finance request failed: %s", exc)
      data = None

    if data is not None and not data.empty:
      break

    if attempt < max_retries:
      wait = backoff_seconds * 2 ** (attempt - 1)
      logger.warning("No data returned; retrying in %.0f seconds", wait)
      time.sleep(wait)
  else:
    logger.error("Giving up on %s after %d attempts",
                 ", ".join(tickerList), max_retries)
    raise RuntimeError(
        f"Could not download data for {tickerList} after {max_retries} attempts")

  # Make sure every ticker actually came back with prices.
  close = data["Close"]
  missing = [t for t in tickerList
             if t not in close.columns or close[t].isna().all()]
  if missing:
    logger.error("No price data for: %s", ", ".join(missing))
    raise ValueError(f"No price data returned for tickers: {missing}")

  # Warn (but continue) when a ticker has gaps in its history.
  for t in tickerList:
    gaps = int(close[t].isna().sum())
    if gaps:
      logger.warning("%s is missing %d of %d closing prices", t, gaps, len(close))

  logger.info("Downloaded %d trading days for %s", len(data), ", ".join(tickerList))
  return data.reset_index()


adata = download_prices("cost aapl nvda","2000-01-01","2023-05-26")

adata["Close","COST"]

adata["percentChange","COST"]=np.concatenate(([0],(adata["Close","COST"].values[1:]-adata["Close","COST"].values[:-1])/adata["Close","COST"].values[:-1]))
adata["percentChange","AAPL"]=np.concatenate(([0],(adata["Close","AAPL"].values[1:]-adata["Close","AAPL"].values[:-1])/adata["Close","AAPL"].values[:-1]))
adata["percentChange","NVDA"]=np.concatenate(([0],(adata["Close","NVDA"].values[1:]-adata["Close","NVDA"].values[:-1])/adata["Close","NVDA"].values[:-1]))
adata
np.cov(adata["percentChange"].T)
adata["percentChange"].std()

def calcPerformance(adata,weights):
  """Annualized return and downside risk for one or more portfolios.

  Args:
    adata: DataFrame containing "percentChange" and "percentChangePositive"
      column groups (daily returns and negative-only daily returns).
    weights: Array of shape (n_portfolios, n_stocks); each row sums to 1.

  Returns:
    (means, stds): annualized mean return and annualized downside deviation,
    one value per portfolio.
  """

  means = ((weights@adata["percentChange"].values.T).mean(axis = 1)+1)**252-1
  stds = (weights@adata["percentChangePositive"].values.T).std(axis = -1) * np.sqrt(252)
  return means, stds

weights = np.random.rand(5,3)
weights = weights/weights.sum(axis=-1,keepdims = True)



def findWeights (adata,stockNames):
  """Pick the portfolio weights with the best Sortino ratio.

  Adds daily return columns for each ticker, generates 5,000 random weight
  vectors, scores them all at once with vectorized matrix math, and returns
  the best one.

  Args:
    adata: Price DataFrame from download_prices (modified in place).
    stockNames: List of tickers, e.g. ["COST", "AAPL", "NVDA"].

  Returns:
    1-D array of weights (one per ticker) that sums to 1.
  """
  for x in stockNames:
    adata["percentChange",x]=np.concatenate(([0],(adata["Close",x].values[1:]-adata["Close",x].values[:-1])/adata["Close",x].values[:-1]))
    adata["percentChangePositive",x]=adata["percentChange",x]
    adata["percentChangePositive",x].values[adata["percentChangePositive",x].values >= 0]=0

  weights = np.random.rand(5000,len(stockNames))
  weights =weights/weights.sum(axis = 1,keepdims = True)
  simulatedMeans, simulatedSTDS = calcPerformance(adata,weights)
  return weights[np.argmax(simulatedMeans/simulatedSTDS)]

y = np.expand_dims(findWeights(adata, ["COST", "AAPL","NVDA"]),axis = 0)
y.shape

calcPerformance(adata,y)

oldadata = download_prices("cost aapl nvda f jnj","2000-01-01","2010-12-31")
newadata = download_prices("cost aapl nvda f jnj","2011-01-01","2023-06-26")

a = np.expand_dims(findWeights(oldadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)
b = np.expand_dims(findWeights(newadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)

a

calcPerformance(newadata,a)

calcPerformance(newadata,b)

((a@newadata["percentChange"].values.T)+1).cumprod()
newadata["portfolioValue"]= ((a@newadata["percentChange"].values.T)+1).cumprod()

((b@newadata["percentChange"].values.T)+1).cumprod()
newadata["portfolioValueIdeal"]= ((b@newadata["percentChange"].values.T)+1).cumprod()
px.line(newadata,x="Date",y = ["portfolioValue","portfolioValueIdeal"])

class portfolio:
  def __init__(self):
    self.startDate = "2000-01-01"

pd.melt(newadata,value_vars = [("portfolioValueIdeal", ""), ("portfolioValue", "")],id_vars= [("Date", "")])

print(newadata.columns )

b

def testing():
  """Walk-forward backtest from 2001 to 2023.

  Each year, weights are chosen using only earlier years' data ("optimized"),
  compared with weights chosen on that same year ("ideal", a hindsight
  benchmark), an equal-weight portfolio, and buy-and-hold of each stock.

  Returns:
    (performance, idealPerformance, perfData, aTotal): yearly (return, risk)
    tuples for the optimized and ideal portfolios, a DataFrame of daily
    portfolio values, and the daily weights used by the optimized portfolio.
  """
  data = download_prices("cost aapl nvda f jnj","2000-01-01","2023-01-01")
  x = 2000
  performance = []
  idealPerformance = []
  perfData = None
  aTotal = None
  bTotal = None
  cTotal = None
  while x < 2023:
    x += 1
    oldadata = data[data.Date.dt.year<x]
    newadata = data[data.Date.dt.year==x]
    a = np.expand_dims(findWeights(oldadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)
    b = np.expand_dims(findWeights(newadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)
    c = np.array([[.2,.2,.2,.2,.2]])
    performance.append(calcPerformance(newadata,a))
    idealPerformance.append(calcPerformance(newadata,b))
    newadata["portfolioValue"]= ((a@newadata["percentChange"].values.T)+1).cumprod()
    newadata["portfolioValueIdeal"]= ((b@newadata["percentChange"].values.T)+1).cumprod()
    a = np.repeat(np.expand_dims(findWeights(oldadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0),newadata.shape[0],axis = 0)
    b = np.repeat(np.expand_dims(findWeights(newadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0),newadata.shape[0],axis = 0)
    c = np.repeat(np.array([[.2,.2,.2,.2,.2]]),newadata.shape[0],axis = 0)


    if perfData is None:
      perfData = newadata
      aTotal = a
      bTotal = b
      cTotal = c
    else:
      perfData = pd.concat([perfData,newadata],axis = 0)
      aTotal = np.concatenate([aTotal,a],axis = 0)
      bTotal = np.concatenate([bTotal,b],axis = 0)
      cTotal = np.concatenate([cTotal,c],axis = 0)
  print(aTotal.shape, perfData["percentChange"].shape)
  print(((aTotal@perfData["percentChange"].values.T)+1).shape)
  print((((aTotal@perfData["percentChange"].values.T)+1).cumprod().T).shape)
  print(perfData["portfolioValue"].shape)
  perfData["portfolioValue"]= ((aTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()
  perfData["portfolioValueIdeal"]= ((bTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()
  perfData["portfolioValueEven"]= ((cTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()
  for ticker in ["AAPL", "NVDA", "COST", "JNJ", "F" ]:
    perfData["portfolioValue" + ticker]= (perfData[("percentChange", ticker)].values+1).cumprod()

  return performance, idealPerformance, perfData,aTotal



performance, idealPerformance, perfData, weights = testing()

print(performance, idealPerformance,)

perfData

px.line(perfData,x="Date", y = ["portfolioValue","portfolioValueEven","portfolioValueAAPL","portfolioValueF","portfolioValueNVDA","portfolioValueCOST","portfolioValueJNJ"])

perfData.columns

s = pd.DataFrame(weights,columns = ["COST", "AAPL","NVDA","F","JNJ"])
s["Date"]= perfData["Date"]
#px.line(s,x="Date",y = ["COST", "AAPL","NVDA","F","JNJ"])
s.groupby(s.Date.dt.year).mean()
px.bar(s.groupby(s.Date.dt.year).mean())

for x in ["portfolioValue", "portfolioValueEven","portfolioValueIdeal"]:
  perfData["percentChange",x]=np.concatenate(([0],(perfData[x].values[1:]-perfData[x].values[:-1])/perfData[x].values[:-1]))
  perfData["percentChangePositive",x]=perfData["percentChange",x]
  perfData["percentChangePositive",x].values[perfData["percentChangePositive",x].values >= 0]=0

perfData["percentChangePositive"].std()* np.sqrt(252)

perfData[["portfolioValue", "portfolioValueEven","portfolioValueIdeal"]].iloc[-1]**(1/23)-1

def testing2():
  """Same walk-forward backtest as testing(), but stores portfolio values
  under the "Close" column group so they can be compared with individual
  stocks on a single risk vs. return scatter plot.

  Returns:
    (performance, idealPerformance, perfData, aTotal), as in testing().
  """
  data = download_prices("cost aapl nvda f jnj","2000-01-01","2023-01-01")
  x = 2000
  performance = []
  idealPerformance = []
  perfData = None
  aTotal = None
  bTotal = None
  cTotal = None
  while x < 2023:
    x += 1
    oldadata = data[data.Date.dt.year<x]
    newadata = data[data.Date.dt.year==x]
    a = np.expand_dims(findWeights(oldadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)
    b = np.expand_dims(findWeights(newadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0)
    c = np.array([[.2,.2,.2,.2,.2]])
    performance.append(calcPerformance(newadata,a))
    idealPerformance.append(calcPerformance(newadata,b))
    newadata["portfolioValue"]= ((a@newadata["percentChange"].values.T)+1).cumprod()
    newadata["portfolioValueIdeal"]= ((b@newadata["percentChange"].values.T)+1).cumprod()
    a = np.repeat(np.expand_dims(findWeights(oldadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0),newadata.shape[0],axis = 0)
    b = np.repeat(np.expand_dims(findWeights(newadata, ["COST", "AAPL","NVDA","F","JNJ"]),axis = 0),newadata.shape[0],axis = 0)
    c = np.repeat(np.array([[.2,.2,.2,.2,.2]]),newadata.shape[0],axis = 0)
    if perfData is None:
      perfData = newadata
      aTotal = a
      bTotal = b
      cTotal = c
    else:
      perfData = pd.concat([perfData,newadata],axis = 0)
      aTotal = np.concatenate([aTotal,a],axis = 0)
      bTotal = np.concatenate([bTotal,b],axis = 0)
      cTotal = np.concatenate([cTotal,c],axis = 0)
  print(aTotal.shape, perfData["percentChange"].shape)
  print(((aTotal@perfData["percentChange"].values.T)+1).shape)
  print((((aTotal@perfData["percentChange"].values.T)+1).cumprod().T).shape)
  print(perfData["portfolioValue"].shape)
  perfData["Close","portfolioValue"]= ((aTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()
  perfData["Close","portfolioValueIdeal"]= ((bTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()
  perfData["Close","portfolioValueEven"]= ((cTotal*perfData["percentChange"].values).sum(axis=1)+1).cumprod()


  return performance, idealPerformance, perfData,aTotal

performance2, idealPerformance2, perfData2, weights2 = testing2()

returns = (((perfData2[["Close"]].iloc[-2])/(perfData2[["Close"]].iloc[0]))**(1/23)-1)

for x in ["portfolioValueEven", "portfolioValue","portfolioValueIdeal","AAPL"]:
  perfData2["percentChange",x]=np.concatenate(([0],(perfData2["Close",x].values[1:]-perfData2["Close",x].values[:-1])/perfData2["Close",x].values[:-1]))
  perfData2["percentChangePositive",x]=perfData2["percentChange",x]
  perfData2["percentChangePositive",x].values[perfData2["percentChangePositive",x].values >= 0]=0

risk =perfData2["percentChangePositive"].std()* np.sqrt(252)

risk = risk.sort_index()
returns = returns.sort_index()

returns.values

risk.index,returns.index

fig = px.scatter(x = risk.values, y = returns.values,color = risk.index, text = risk.index)
fig.update_layout(plot_bgcolor = 'white')
fig.update_xaxes(showline = True, linecolor = "black")
fig.update_yaxes(showline = True,linecolor = "black")
fig.show()

returns