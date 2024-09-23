import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('MNIST_100.csv')
y = data.iloc[:, 0]
x = data.drop('label', axis=1)
#print(x)
pca = PCA(n_components=2)
pca.fit(x)
PCAX = pca.transform(x)
plt.scatter(PCAX[0:100, 0], PCAX[0:100, 1])
plt.scatter(PCAX[100:200, 0], PCAX[100:200, 1])
plt.scatter(PCAX[200:300, 0], PCAX[200:300, 1])
plt.scatter(PCAX[300:400, 0], PCAX[300:400, 1])
plt.scatter(PCAX[400:500, 0], PCAX[400:500, 1])
plt.scatter(PCAX[500:600, 0], PCAX[500:600, 1])
plt.scatter(PCAX[600:700, 0], PCAX[600:700, 1])
plt.scatter(PCAX[700:800, 0], PCAX[700:800, 1])
plt.scatter(PCAX[800:900, 0], PCAX[800:900, 1])
plt.scatter(PCAX[900:1000, 0], PCAX[900:1000, 1])

plt.show()
