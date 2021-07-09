# %%
import pickle
import numpy as np

# %%
real_points = np.reshape(np.array(pickle.load(open("D:/Personal Projects/irsdk_hybrid_util/test_features.p", 'rb'))), (150, 8))
test_points = np.array(pickle.load(open("D:/Personal Projects/irsdk_hybrid_util/test_points.p", 'rb')))[:, 1:]

# %%
