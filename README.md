# Portfolio Optimizer

A Python project that finds stock portfolio weights with the best risk-adjusted return, then backtests that strategy against an equal-weight portfolio and individual stocks.

Originally built in Google Colab.

## What it does

1. **Downloads historical prices** for a set of stocks (COST, AAPL, NVDA, F, JNJ) from Yahoo Finance using `yfinance`.
2. **Computes daily returns** for each stock.
3. **Optimizes portfolio weights** by Monte Carlo search. It generates 5,000 random weight combinations and picks the one with the highest **Sortino ratio**, which is annualized return divided by annualized *downside* volatility. Only negative days count as risk.
4. **Runs a walk-forward backtest (2001–2022).** Each year, it chooses weights using only data from earlier years, then holds those weights for the year. This avoids look-ahead bias.
5. **Compares strategies:**
   - **Optimized:** weights chosen from past data only
   - **Ideal (hindsight):** weights chosen using that same year's data, as an upper-bound benchmark
   - **Equal-weight:** 20% in each stock
   - **Each individual stock:** buy and hold
6. **Visualizes results** with Plotly:
   - Portfolio growth over time
   - Average yearly weight given to each stock
   - A risk vs. return scatter plot of every strategy

## Key functions

| Function | Purpose |
|---|---|
| `calcPerformance(adata, weights)` | Returns annualized mean return and annualized downside deviation for one or more weight sets. |
| `findWeights(adata, stockNames)` | Computes daily returns for the given tickers and returns the weight vector with the best Sortino ratio out of 5,000 random portfolios. |
| `testing()` | Runs the year-by-year walk-forward backtest and builds portfolio value series for each strategy. |
| `testing2()` | Same backtest, but stores results so returns and risk can be compared on a single scatter plot. |

## Requirements

- Python 3.9+
- pandas
- numpy
- yfinance
- plotly
- scipy

Install them with:

```bash
pip install pandas numpy yfinance plotly scipy
```

## How to run

**Option 1: Google Colab (recommended).** Upload `PortfolioOptimizer.py` to Colab, or paste it into a notebook, and run the cells in order. Plotly charts display inline.

**Option 2: Locally.**

```bash
python PortfolioOptimizer.py
```

The code was written as notebook cells, so some charts made with `px.line(...)` or `px.bar(...)` only show up in a notebook. To display them when running as a script, add `.show()` to those calls.

## Customizing

- **Change the stocks:** edit the ticker string in `yf.download(...)` and the `stockNames` list passed to `findWeights`.
- **Change the date range:** edit the start and end dates in `yf.download(...)`.
- **Change the search size:** the number of random portfolios is set in `findWeights` with `np.random.rand(5000, ...)`.

## Notes and limitations

- Results change slightly on every run because the weight search is random.
- No transaction costs, taxes, or dividends beyond adjusted close prices are included.
- The "Ideal" portfolio uses future information and is only a benchmark. It is not a tradable strategy.
- This is an educational project, **not financial advice**.
