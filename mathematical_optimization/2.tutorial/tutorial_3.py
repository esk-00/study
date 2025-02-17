import pandas as pd
import pulp

stock_df = pd.read_csv("stocks.csv")
print(stock_df)

require = pd.read_csv("requires.csv")
print(require)

gain_df = pd.read_csv("gains.csv")
print(gain_df)
