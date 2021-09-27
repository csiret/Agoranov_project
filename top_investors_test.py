from pickle import TRUE
from matplotlib import lines
from numpy.core.defchararray import center
from numpy.lib.function_base import average
from tqdm.utils import _supports_unicode
from herding_measures import average_velocity
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from herding_measures import lead_cos_herding
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D

df = pd.read_csv("investor_profile.csv")
community_matrices = np.load("investor_profile_matrices.npy")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
investment_counter = np.load("investment_counter.npy")
community_matrices = community_matrices[:, :, 40:78, :]

#excracting top 30% of investors for every community per leading measures
time_step = 3
top_investors = np.zeros(shape=(int((community_matrices.shape[1])/2)+1, community_matrices.shape[3]))
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
    sorted_data = sorted_data[~np.isnan(sorted_data)]
    for j in range(int((investor_no)/2)+1):
        top_investors[j, i] = np.where(data[:, i] == sorted_data[j])[0][0]

counter = 0
average_velocity_data = np.zeros(shape=(community_matrices.shape[2], community_matrices.shape[3], 4))
for t_sig in [4, 12, 20, 60]:
    for i in range(community_matrices.shape[3]):
        top_community_investors = top_investors[:,i][top_investors[:,i] != 0].astype(int)
        community = community_matrices[:, top_community_investors, :, i]
        investor_no = len(top_community_investors)
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

fig.suptitle("Average velocities per community for top investor systems (25%)")
plt.setp(axes , xticks = x, xticklabels = tags)
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fig.legend(average_velocity_data, labels = labels)
plt.show()