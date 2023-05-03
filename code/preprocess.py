import pandas as pd
import numpy as np

# Preprocess funciton:
# 1. Removes null and infinity values
# 2. Creates new variables workday and day/night
# 3. One-hot encodes ltecell_name
# 4. Drops given numerical columns
# 5. Returns a dataframe

def preprocess(data, drop_cols):

    #Remove nulls and infs
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    #Create time columns
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.weekday

    #Create new features
    df['day/night'] = [1 if 7 < x else 0 for x in df['hour']]
    df['workday'] = [1 if 4 < x else 0 for x in df['day']]

    #One-Hot encode different cells
    dummies = pd.get_dummies(df['ltecell_name'])
    df_dummies = pd.concat([df, dummies], axis=1)

    #Drop selected columns
    final_df = df_dummies.drop(drop_cols, axis=1)

    return final_df