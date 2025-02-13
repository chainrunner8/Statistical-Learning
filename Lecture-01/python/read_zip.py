import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Prepare training data
train = pd.read_csv("zip.train", sep = " ", header=None)
print(train.shape)
print(train.isna().sum().sum()) # last columns has only missings
train = train.drop([257], axis=1)

## Prepare test data
test = pd.read_csv("zip.test", sep = " ", header=None)
print(test.isna().sum().sum())
print(test.shape)## V1 is an indicator for the digit, select 2s and 3s
train = train[(train[0] >= 2) & (train[0] <= 3)]
train[0] = train[0] - 2
print(train.shape)
test = test[(test[0] >= 2) & (test[0] <= 3)]
test[0] = test[0] - 2
print(test.shape)

## This is what the images look like:
for i in range(10):
  im = np.array(train[train[0] == i][list(range(2, 258))].iloc[0, :]).reshape(16, 16)
  plt.imshow(np.fliplr(np.rot90(im, 1)), cmap='gray', vmin=0, vmax=63/63)
  plt.axis('off')
  plt.show()
  ## 63 is the minimum number of unique grayscale values observed for any variable 
