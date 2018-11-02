import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/San Francisco-listings.csv')

def dollartofloat(s):
    s = s.replace('$', '')
    s = s.replace(',', '')
    return float(s)

prices = np.array([dollartofloat(d) for d in df['price'].values])

print("mean: ", np.mean(prices))
print("max: ", np.max(prices))
print("min: ", np.min(prices))

plt.hist(prices, bins=100)
plt.show()