from pickle import TRUE
from matplotlib import lines
from numpy.core.defchararray import center
from numpy.lib.function_base import average
from herding_measures import average_velocity
from numpy.core.fromnumeric import shape
from herding_measures import herding
import pandas as pd
import numpy as np
from herding_measures import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D

community_matrices = np.load("investor_profile_matrices.npy")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
investment_counter = np.load("investment_counter.npy")
community_matrices = community_matrices[:, :, 40:, :]

#plotting average velocities for all communities at all times after 2010 for different time steps
counter = 0
average_velocity_data = np.zeros(shape=(community_matrices.shape[2], community_matrices.shape[3], 4))
for t_sig in [1, 4, 12, 20]:
    for i in range(community_matrices.shape[3]):
        community = community_matrices[:, :, :, i]
        community_investors = community_df.iloc[:, i]
        community_investors.dropna(inplace=True)
        investor_no = community_investors.shape[0]
        for t in range(community_matrices.shape[2]):
            average_velocity_data[t, i, counter] = average_velocity(community, t, t_sig, investor_no)
    counter += 1

plt.style.use("seaborn")

tags = [2010, 2015, 2020]
x = [0, 20, 40]
fig, axes = plt.subplots(2,2)
titles = ["Time-step = 1 year", "Time-step = 3 years", "Time-step = 5 years", "No time-step"]
for row_num in range(2):
    for col_num in range(2):
        ax = axes[row_num][col_num]
        ax.plot(average_velocity_data[:, :, 2*row_num+col_num])
        ax.set_title(titles[2*row_num+col_num])

fig.suptitle("Average velocities per community for different time-steps")
plt.setp(axes , xticks = x, xticklabels = tags)
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fig.legend(average_velocity_data, labels = labels)
plt.show()