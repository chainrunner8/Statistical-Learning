from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets

