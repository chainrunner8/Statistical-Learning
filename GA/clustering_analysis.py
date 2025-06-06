import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_X = PCA()
pca_X.fit(X_scaled)
scores = pca_X.transform(X_scaled)


