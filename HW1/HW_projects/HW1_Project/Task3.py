import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('housing_training.csv', header=None)
x = data.iloc[:, 0]
print(x)
plt.hist(x)
plt.show()