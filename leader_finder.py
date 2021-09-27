from numpy.core.arrayprint import DatetimeFormat
from numpy.core.defchararray import index
from numpy.lib.function_base import average
from herding_measures import average_velocity
from numpy.core.fromnumeric import _searchsorted_dispatcher, shape
from herding_measures import herding
import pandas as pd
import numpy as np
from herding_measures import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

community_matrices = np.load("investor_profile_matrices.npy")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
community_matrices = community_matrices[:, :, 40:77, :]

#finding investors with highest leading measures
leaders = np.zeros(shape=(community_matrices.shape[3], 6, len(range(1,11))))
for time_step in tqdm(range(10)):
    data = np.zeros(shape=(community_matrices.shape[1], community_matrices.shape[3]))
    for i in range(community_matrices.shape[3]):
        community = community_matrices[:, :, :, i] 
        community_investors = community_df.iloc[:, i]
        community_investors.dropna(inplace=True)
        investor_no = community_investors.shape[0]    
        for investor in range(investor_no):
            leading_measures = np.zeros(shape=(community.shape[2]-(time_step+1)))
            for t in range(time_step+1, community.shape[2]):
                leading_measures[t-(time_step+1)] = lead_cos_herding(investor, community, t, time_step+1)
            data[investor, i] = np.average(leading_measures[~np.isnan(leading_measures)])
        sorted_data = np.sort(data[:,i])[::-1]
        leaders[i, 0, time_step] = np.where(data[:, i] == np.amax(data[:, i]))[0]
        leaders[i, 1, time_step] = np.amax(data[:, i])
        leaders[i, 2, time_step] = np.where(data[:,i] == sorted_data[1])[0]
        leaders[i, 3, time_step] = sorted_data[1]
        leaders[i, 4, time_step] = np.where(data[:,i] == sorted_data[2])[0]
        leaders[i, 5, time_step] = sorted_data[2]

np.save("leaders", leaders)
  
