import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("datasets/homeprices.csv")

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

x = np.linspace(2400, 6000, 5000)
y = reg.coef_*x+reg.intercept_

test_data = pd.read_csv("datasets/areas_test.csv")

pred = reg.predict(test_data)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(x, y, color='blue')
plt.scatter(test_data, pred, color='orange')
plt.show()
