

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

"""Construct investor location vectors and store in time-sequence matrices"""

#read data
df = pd.read_csv("investor_profile.csv")
community_df = pd.read_csv("communities_990_draw.csv", index_col=0)

#set up dimensions, sector_groups and sectors
sector_group_list = [str(i).split(",") for i in df.sector_groups.unique()]
sector_group_list = [item for sublist in sector_group_list for item in sublist]
sector_group_list = np.unique(np.array(sector_group_list))
sector_group_list = np.delete(sector_group_list, 41)

#set up investor portfolios
investor_list = pd.unique(df[["investor_permalink_orgs", "investor_permalink_people"]].values.ravel('K'))
investor_matrices = np.zeros([len(sector_group_list), 183, 4*len(range(2000, 2022)), community_df.shape[1]])
df["date"] = pd.to_datetime(df["date"])
df = df[df.date.dt.year >= 2000]

#looping over communities and investors
for community_no in tqdm(range(community_df.shape[1])):
    community = community_df.iloc[:, community_no]
    community.dropna(inplace=True)
    for investor in community.values:
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
            
                #adding information to investor profile matrix
                investor_matrices[np.where(sector_group_list == sector),
                                np.where(np.array(community.values) == investor),
                                (current_row.date.dt.year-2000)*4 + int((current_row.date.dt.month-1)/3),
                                community_no] += 1/sqrt(n)

np.save("investor_profile_matrices", investor_matrices)
