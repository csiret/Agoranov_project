import pandas as pd
import numpy as np

community_df = pd.read_csv("communities_990_draw.csv")

#replace incorrect permalinks
community_df.replace({"acceleprise-ventures":"forumventures", "e-ventures":"headlinevc",
                      "point-judith-capital":"pjc", "capdecisif-management":"karista",
                      "Investiere":"verve-ventures", "zillionize-angel":"zillionize"}, value=None, inplace=True)

#save data
community_df.to_csv(r"C:\Users\charl\Documents\Agoranov_data\bulk_export_19-07-2021\communities_990_draw.csv")
