from herding_measures import threedee, herding_measures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

community_df = pd.read_csv("communities_990_draw.csv")
all_communities = np.load("3D_model_matrix.npy")[:, 40:82, :]
array = np.reshape(all_communities, (all_communities.shape[0]*all_communities.shape[1], all_communities.shape[2]))
community = np.load("investor_profile_matrices.npy")[:, 40:82, :, 2]
np.transpose(community, (1, 2, 0))
investors = community_df.iloc[:, 2]
investors.dropna(inplace=True)
investor_list = investors.values
community = community[:len(investor_list), :, :]
threedee.plot_embeds(threedee.run_global_umap(threedee.fit_umap(array), community))

plt.show()
