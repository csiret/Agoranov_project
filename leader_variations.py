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

#finding mean and standard deviation of similarity of an investor to himself for all leaders and with different time steps
leader_variations = np.zeros(shape=(leaders.shape[0], 9, len(range(1,11))))
for time_step in range(1,11):
    for i in range(community_matrices.shape[3]):
        community = community_matrices[:, :, :, i]
        current_leaders = leaders[i, :, 3]
        for j in range(3):
            current_leader = int(current_leaders[2*j])
            variations = np.zeros(shape=(community_matrices.shape[2]-time_step))
            for t in range(time_step, community_matrices.shape[2]):
                variations[t-time_step] = self_similarity(community, current_leader, t, time_step)
            leader_variations[i, 3*j, time_step-1] = current_leader
            leader_variations[i, 3*j + 1, time_step-1] = np.mean(variations[~np.isnan(variations)])
            leader_variations[i, 3*j + 2, time_step-1] = np.std(variations[~np.isnan(variations)])

#plotting similarity distributions
time_step = 2
distributions = np.zeros(shape=(community_matrices.shape[2] - time_step, 3, leaders.shape[0]))
for i in range(community_matrices.shape[3]):
    community = community_matrices[:, :, :, i]
    current_leaders = leaders[i, :, 3]
    for j in range(3):
        current_leader = int(current_leaders[2*j])
        for t in range(time_step, community_matrices.shape[2]):
            distributions[t-time_step, j, i] = self_similarity(community, current_leader, t, time_step)

plt.style.use("seaborn")
tags = [2010, 2013, 2015, 2018]
x = [0, 12, 20, 32]
fig, axes = plt.subplots(5,2)
titles = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
for row_num in range(5):
    for col_num in range(2):
        ax = axes[row_num][col_num]
        ax.plot(distributions[:, :, 2*row_num+col_num])
        ax.set_title(titles[2*row_num+col_num])

fig.suptitle("Leader investment variations for time step = 2 quarters")
plt.setp(axes , xticks = x, xticklabels = tags)
labels = [1, 2, 3]
fig.legend(distributions, labels = labels)
summary_stats = pd.DataFrame(distributions[:, :, 0]).describe()

plt.show()


