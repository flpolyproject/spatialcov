import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()



flights = sns.load_dataset("flights")
print(flights)
flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights)
plt.show()

print(flights)