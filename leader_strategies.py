from datetime import time
from herding_measures.herding_measures import self_difference
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
from herding_measures.herding_measures import self_similarity

community_matrices = np.load("investor_profile_matrices.npy")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
community_matrices = community_matrices[:, :, 40:78, :]
leaders = np.load("leaders.npy")
investment_counter = np.load("investment_counter.npy")

#finding average trends across all investors
investment_counter = investment_counter[:, :, 40:82, :]
average_trends = np.zeros(shape=(investment_counter.shape[0] , len(range(1,11)), investment_counter.shape[3]))
for time_step in range(1,11):
    for i in range(community_matrices.shape[3]):
        community = investment_counter[:, :, :, i]
        investor_list = community_df.iloc[:, i]
        investor_list.dropna(inplace=True)
        investment_differences = np.zeros(shape=(investment_counter.shape[0], len(investor_list.values)))
        for investor in range(len(investor_list.values)):
            variations = np.zeros(shape=(community.shape[0], community.shape[2]-time_step))
            for t in range(time_step, community_matrices.shape[2]):
                variations[:, t-time_step] = self_difference(community, investor, t, time_step)
            investment_differences[:, investor] = np.average(variations, axis = 1)
        average_trends[:, time_step-1, i] = np.average(investment_differences, axis=1)

#finding average trends for leaders
investment_differences = np.zeros(shape=(investment_counter.shape[0], leaders.shape[0]*3 , len(range(1,11))))
for time_step in range(1,11):
    for i in range(community_matrices.shape[3]):
        community = investment_counter[:, :, :, i]
        current_leaders = leaders[i, :, 3]
        for j in range(3):
            current_leader = int(current_leaders[2*j])
            variations = np.zeros(shape=(investment_counter.shape[0], community_matrices.shape[2]-time_step))
            for t in range(time_step, community_matrices.shape[2]):
                variations[:, t-time_step] = self_difference(community, current_leader, t, time_step)
            investment_differences[:, 3*i + j, time_step-1] = np.average(variations, axis = 1)

        