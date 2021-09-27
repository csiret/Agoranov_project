from numpy.core.defchararray import index
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np


"""Removing permalinks that are not in investor profile dataframe"""


#read data
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
df = pd.read_csv("investor_profile.csv")

#remove investors not found in investor_profile dataframe
investor_list = pd.unique(df[["investor_permalink_orgs", "investor_permalink_people"]].values.ravel('K'))
removed_investor_list = []
for i in range(community_df.shape[1]):
    permalink_list = community_df.pop(f"{i}")
    for j in range(len(permalink_list)):
        if permalink_list[j] not in investor_list:
            removed_investor_list.append(permalink_list[j])
            del permalink_list[j]
    community_df[f"{i}"] = permalink_list
print(removed_investor_list)

#save modified dataframe
community_df.to_csv(r"C:\Users\charl\Documents\Agoranov_data\bulk_export_19-07-2021\communities_990_draw.csv")

