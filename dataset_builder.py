

from numpy.core.numeric import outer
import pandas as pd

"""Construct investor profile dataframe using merge()"""

#read data
df_fund_round = pd.read_csv("funding_rounds.csv")
df_investments = pd.read_csv("investments.csv")
df_org = pd.read_csv("organizations.csv")
df_people = pd.read_csv("people.csv")

#start from investments dataframe
df = df_investments

#renaming columns to match investments dataframe/dropping unnecessary columns
df_fund_round.drop(["rank", "country_code", "state_code", "created_at",
                    "updated_at", "region", "city", "type"], axis = 1,
                     inplace = True)
df_fund_round.rename(columns={"uuid" : "funding_round_uuid",
                              "name" : "funding_round_name",
                              "announced_on" : "date"},
                              inplace = True)

#merge investments and funding round dataframe to obtain date,
#amount raised, organization name and organization uuid
df = pd.merge(df, df_fund_round, how="left", on=["funding_round_uuid",
              "funding_round_name"])

#renaming columns and drop unneccessary information
df_org.drop(["type", "rank", "cb_url", "created_at",
             "updated_at", "roles", "domain", "homepage_url", "country_code",
             "state_code", "region", "city", "address", "postal_code",
             "status", "short_description", "email", "phone",
             "facebook_url", "linkedin_url", "twitter_url",
             "logo_url", "alias1", "alias2", "alias3"], axis=1,
             inplace=True)
df_org.rename(columns={"uuid": "org_uuid", "name" : "org_name",
                       "category_groups_list" : "sector_groups",
                       "category_list" : "sector"},
                       inplace=True)

#merge dataframe with organization dataframe to obtain sector
#information
df = pd.merge(df, df_org, how="left", on=["org_uuid", "org_name"])
df.drop(["permalink"], axis=1, inplace=True)

#extract investor permalinks from organization dataframe
df_org.drop(["sector_groups", "sector", "org_name"], axis=1, inplace=True)
df_org.rename(columns={"org_uuid" : "investor_uuid",
                       "permalink": "investor_permalink_orgs"}, inplace=True)
df = pd.merge(df, df_org, how = "left", on = ["investor_uuid"])

#dropping unnecessary information from people dataframe 
df_people.drop(["name", "type", "cb_url", "rank", "created_at", "updated_at",
                "first_name", "last_name", "gender", "country_code", "state_code",
                "region", "city", "featured_job_organization_uuid", "featured_job_organization_name",
                "facebook_url", "linkedin_url", "twitter_url", "logo_url"], axis=1, inplace=True)
df_people.rename(columns={"uuid" : "investor_uuid",
                          "permalink" : "investor_permalink_people"}, inplace=True)

#merge people dataframe to obtain people investor uuids
df = pd.merge(df, df_people, how = "left", on = ["investor_uuid"])

#sort data by date
df["date"] = pd.to_datetime(df["date"])
df.sort_values(by="date", inplace=True)

#save data
df.to_csv(r"C:\Users\charl\Documents\Agoranov_data\bulk_export_19-07-2021\investor_profile.csv")

