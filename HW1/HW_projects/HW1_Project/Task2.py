import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('housing_training.csv', header=None)
x = data.iloc[:, [10,12,13]]
print(x)
plt.violinplot(x)
plt.show()
