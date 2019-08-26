import pandas as pd
import numpy as np
import sys

instrument = sys.argv[1]
data_file = "./data/" + instrument + "HOT_1m.csv"
dim = 240

### set index to date ###
df = pd.read_csv(data_file)
df['date'] = df[['time']].apply(lambda x: x//1000000)
date_list = ['-'.join([str(d)[:4],str(d)[4:6],str(d)[6:8]])for d in df['date']]
df['date'] = date_list
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[['high','low','open','close']]

### format data ###
grouped_df = df.groupby('date')
for key, group in grouped_df:
    if (group.count()[0] < dim): # remove outliers
        df = df.drop(key)
# select the first 240 out of 270 records per day before 2016
df = df.groupby('date').head(dim)

### transform dataframe into numpy array ###
size = len(df.groupby('date')) # IC-988 IH-890 IF-2213
# print(size)
arr = np.zeros((size, dim, 4), dtype=np.float)
final_grouped_df = df.groupby('date')

cnt = 0
for key, group in final_grouped_df:
    arr[cnt] = group.values
    cnt += 1

np.save("./data/"+instrument+".npy", arr)