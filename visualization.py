import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
import matplotlib.gridspec as gridspec # 新增导入gridspec用于分割子图
from pandas import DataFrame
from data_loader import load_data

def load_plot_data(data_file, start, end):
    df_stockload = pd.read_csv(data_file)[['instrument','time','high','low','open','close']]
    df_stockload['date'] = df_stockload[['time']].apply(lambda x: x//1000000)
    date_list = ['-'.join([str(d)[:4],str(d)[4:6],str(d)[6:8]])for d in df_stockload['date']]
    df_stockload['date'] = date_list
    df_stockload['date'] = pd.to_datetime(df_stockload['date'])
    df_stockload = df_stockload.set_index('date')
    df_stockload = df_stockload.loc[start:end]
    # print(df_stockload.head())
    return df_stockload

def plot(df, epoch=0, num_batch=0, i=0, instrument=None, generate=False, k=True, v=True):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(figsize=(8,6), dpi=100,facecolor="white") # 创建fig对象
    gs = gridspec.GridSpec(2, 1, left=0.05, bottom=0.15, right=0.96, top=0.96, wspace=None, hspace=0, height_ratios=[3.5,1])
    graph_KAV = fig.add_subplot(gs[0,:])
    if v:
        graph_VOL = fig.add_subplot(gs[1,:])    

    """ 绘制K线图 """
    if k:
        mpf.candlestick2_ochl(graph_KAV, df.open, df.close, df.high, df.low, width=0.5,
                            colorup='r', colordown='g') # 绘制K线走势
        if instrument == None:
            graph_KAV.set_title(df.instrument[0])
        else:
            graph_KAV.set_title(instrument)
        graph_KAV.set_ylabel("Price")
        graph_KAV.grid(True, color='k')
        graph_KAV.set_xlim(0, len(df.index)) # 设置一下x轴的范围
        graph_KAV.set_xlabel("Minute")

    """ 绘制成交量图 """
    if v:
        numt = np.arange(0, len(df.index))
        graph_VOL.bar(numt, df.volume,color=['g' if df.open[x] > df.close[x] else 'r' for x in range(0,len(df.index))])
        graph_VOL.set_ylabel(u"Volume")
        graph_VOL.set_xlim(0,len(df.index)) # 设置一下X轴的范围
        if not generate:
            graph_VOL.set_xlabel(u"Date")
            graph_VOL.set_xticks(range(0,len(df.index),270)) # X轴刻度设定，每1天标一个日期
            graph_VOL.set_xticklabels([df.index.strftime('%Y-%m-%d')[index] for index in graph_VOL.get_xticks()]) # 标签设置为日期
        # 将日K线X轴labels隐藏
        for label in graph_KAV.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in graph_VOL.xaxis.get_ticklabels():
            label.set_fontsize(10) # 设置标签字号

    if not generate:
        plt.show()
    else:
        if not os.path.exists("./plots/"+instrument+"/epoch_"+str(epoch)):
            os.makedirs("./plots/"+instrument+"/epoch_"+str(epoch))
        plt.savefig("./plots/"+instrument+"/epoch_"+str(epoch)+"/batch_"+str(num_batch)+"_"+str(i)+".png")
        plt.close()

if __name__ == "__main__":
    data_file = "./data/IFHot_1m.csv"
    start = '20160105' # 开始日期
    end = '20160105' # 结束日期

    df = load_plot_data(data_file, start, end)
    data_loader = load_data('IF', 4)
    real_data = next(iter(data_loader))
    dimage = real_data.data[:1][0].numpy()
    df = DataFrame(dimage, columns=['high','low','open','close'])
    plot(df, instrument='IF', generate=True, v=False)

    plot(df, v=False)