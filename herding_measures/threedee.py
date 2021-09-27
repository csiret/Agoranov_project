import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

def get_array(df, complete = True, date = 1998) :
    if not complete :
        df = df[df["year"] == str(date)]
        
    #drop the date column
    array = df.to_numpy()[:, 1:]
    return array

def fit_umap(portfolios, n_components = 3) :
    
    umap_transformer = umap.UMAP(n_neighbors = 20,
                                n_components = n_components,
                                metric = 'cosine',
                                verbose = False).fit(portfolios)
    
    return umap_transformer

def run_global_umap(umap_transformer, all_coms) :
    list_embeds = []
    for date in range(all_coms.shape[1]):
        print(date)
        quarter_array = all_coms[:, date, :]
        quarter_array = np.mean(quarter_array, axis = 0)
        umap_embeds = umap_transformer.transform(quarter_array.reshape(1, -1))
        list_embeds.append(umap_embeds)
    return list_embeds
    
def plot_embeds(list_embeds) :
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax = fig.add_subplot()
    
    for i in range(42):
        embed = list_embeds[i][0]
        ax.scatter(embed[0], embed[1], embed[2], marker = 'o', color = 'r')
        ax.text(embed[0], embed[1], embed[2], str(i), fontsize = 'xx-small')
        # ax.scatter(embed[0], embed[1], marker = 'o', color = 'r')
        # ax.text(embed[0], embed[1], str(date))
        
    return fig, ax

def plot_individual_investor(df, umap_transformer, fig, ax, investor, color = 'b') :
    investor_df = df.loc[investor]
    for i in range(len(investor_df)) :
        date = investor_df.iloc[i]["year"]
        if date == "2020" :
            continue
        array = investor_df.iloc[i].to_numpy()[1:]
        umap_embed = umap_transformer.transform(array.reshape(1, -1))[0]
        ax.scatter(umap_embed[0], umap_embed[1], umap_embed[2], marker = '^', color = color)
        ax.text(umap_embed[0], umap_embed[1], umap_embed[2], str(date), fontsize = 'xx-small')
        # ax.scatter(umap_embed[0], umap_embed[1], marker = '^', color = color)
        # ax.text(umap_embed[0], umap_embed[1], str(date))

    return fig, ax