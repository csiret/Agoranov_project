from numpy.core.fromnumeric import shape
from numpy.core.numeric import NaN
from numpy.lib.mixins import _inplace_binary_method
from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
from pandas.core.indexes.base import Index
import datetime as dt
from tqdm import tqdm
from math import sqrt

df = pd.read_csv("investor_profile.csv")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)
df["date"] = pd.to_datetime(df["date"])
df = df[df.date.dt.year >= 2000]

#set up dimensions, sector_groups and sectors
sector_group_list = [str(i).split(",") for i in df.sector_groups.unique()]
sector_group_list = [item for sublist in sector_group_list for item in sublist]
sector_group_list = np.unique(np.array(sector_group_list))
sector_group_list = np.delete(sector_group_list, 41)

#set up investor portfolios
investor_list = []
for community_no in range(0, community_df.shape[1]):
    community = community_df.iloc[:, community_no]
    community.dropna(inplace=True)
    investor_list.append(community.values.tolist())
investor_list = [item for sublist in investor_list for item in sublist]
    
matrices = np.zeros([len(investor_list), 4*len(range(2000, 2022)), len(sector_group_list)])
for investor in tqdm(investor_list):
    current_df = df[(df.investor_permalink_orgs == investor) | (df.investor_permalink_people == investor)]
        
    #looping over rows in resulting dataframe and extracting
    #investor uuid, and sectors
    for i in range(current_df.shape[0]):
        current_row = current_df.iloc[[i]]
        sectors = [str(current_row["sector_groups"].values[0]).split(",")]
        sectors = [item for sublist in sectors for item in sublist]
        if "Software" in sectors:
            sectors.remove("Software")
        n = len(sectors)
        for sector in sectors:
            matrices[investor_list.index(investor),
                    (current_row.date.dt.year-2000)*4 + int((current_row.date.dt.month-1)/3),
                    np.where(sector_group_list == sector)] += 1/sqrt(n)

np.save("3D_model_matrix", matrices)
