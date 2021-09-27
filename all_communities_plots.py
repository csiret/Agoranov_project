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

all_communities = np.load("3D_model_matrix")

#plotting average velocities for all investors disregarding communities
counter = 0
average_velocity_data = np.zeros(shape=(all_communities.shape[2], 4))
for t_sig in [4, 12, 20, 90]:
    for t in range(all_communities.shape[2]):
        average_velocity_data[t, counter] = average_velocity(all_communities, t, t_sig, all_communities.shape[1]*all_communities.shape[3])
    counter += 1

plt.style.use("seaborn")

tags = [2000, 2005, 2010, 2015, 2020]
x = [0, 20, 40, 60, 80]
fig, axes = plt.subplots(2,2, constrained_layout = True)
titles = ["Time-step = 1 year", "Time-step = 3 years", "Time-step = 5 years", "No time-step"]
for row_num in range(2):
    for col_num in range(2):
        ax = axes[row_num][col_num]
        ax.plot(average_velocity_data[:, 2*row_num+col_num])
        ax.set_title(titles[2*row_num+col_num])

fig.suptitle("Average velocities for all communities for different time-steps")
plt.setp(axes , xticks = x, xticklabels = tags)
ax.legend()
plt.show()
