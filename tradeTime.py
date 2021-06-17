# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:32:01 2021

@author: yuyou
"""

import pandas as pd
import numpy as np
import dolphindb as ddb
import matplotlib.pyplot as plt

#%% # download data from dolphindb
s = ddb.session()
s.connect('10.0.60.55',8508)
s.login('reader','888888')

db_etf = s.loadTable(dbPath = 'dfs://STOCK_SH_TRDMIN', tableName = 'STOCK_SH_TRDMIN')
db_optionSymbol = s.loadTable(dbPath = 'dfs://G2OptionTickStatMinute', tableName = 'TickStat')
db_optionPrices = s.loadTable(dbPath = 'dfs://SHSOL1_TRDMIN_3', tableName = 'SHSOL1_TRDMIN')

#%%  #  get_data_from_db(db_table, '510300', cols, [])
def get_data_from_db(db_table, conditions=["symbol='510300'"], columns=['*']):
    
    temp = db_table.where(conditions).select(columns)
    return temp.toDF()


#%%  # add_timeStr(df)  df为数据表
def add_timeStr(df):
    time_group = []
    df_ = df.reset_index(drop=True)
    for i in range(len(df_)):
        d = df_.tradingDay[i]
        t = df_.time[i]
        tempTime = pd.Timestamp(d.year, d.month, d.day, 
                                t.hour, t.minute, t.second)
        timeStr = tempTime.strftime('%Y.%m.%d %H:%M:%S')
        time_group.append(timeStr)
    df_['time_'] = time_group
    temp = df_.sort_values('time_')
    result = temp.set_index('time_',drop=True)
    
    return result


#%%  # get etf table
def get_etf(db_table, symbol, start, end):
    cons=["symbol='{}'".format(symbol), 
          "tradingDay>={}".format(start.split()[0]), 
          "tradingDay<={}".format(end.split()[0]), 
          "time!=09:30:00", 
          "cycle=1"]
    cols=['symbol','tradingDay','time','open','close','high','low']
    etf = get_data_from_db(db_table, cons, cols)
    etf = add_timeStr(etf)
    res = etf[etf.index<end]
    return res

#%%
def get_tradeTime(df_records, start, end):
    cols = df_records.columns
    tradeRecords = []
    for i in np.arange(0,len(cols),3):
        temp = df_records[cols[i:(i+3)]].dropna()
        times = []
        for j in range(len(temp)):
            dateStr = str(int(temp.iloc[j][cols[i]]))  # 每3列为一组，第i列为date，第i+1列为time，第i+2列为close
            date = '{}.{}.{}'.format(dateStr[0:4],dateStr[4:6],dateStr[6:])
            timeStr = str(int(temp.iloc[j][cols[i+1]])).zfill(4)
            Time = '{}:{}:00'.format(timeStr[0:2],timeStr[2:4])
            
            time_ = '{} {}'.format(date,Time)
            if (time_>=start) & (time_<=end):
                times.append([time_,temp.iloc[j][cols[i+2]].round(4)])
        
        trades = pd.DataFrame(times, columns=['{}'.format(int(i/3)+1), '{}{}'.format(int(i/3)+1,int(i/3)+1)])
        tradeRecords.append(trades)
    
    result = pd.concat(tradeRecords, axis=1)
    
    return result

#%%
def run_everyTrade(C_, P_, strikeC, strikeP, cash=100000.0, holding=0, load=1.0, 
                   toLong=True, start='2019', end='2022', multi=10000):
    
    underly = C_.underlying[0]
    codeC = C_.symbol[0]
    codeP = P_.symbol[0]
    strike = (strikeC+strikeP)*0.5
    dfC = C_[C_.index>=start]
    dfP = P_.copy()
    asset = cash
    
    asset_list = []
    trade_list = []
    
    callValue = 0.0; putValue = 0.0; callClear = 0.0; putClear = 0.0
    signal = ''
    
    for i in range(len(dfC)):

        closeC = dfC['close'][i]
        closeP = dfP['close'][i]
        
        if dfC.index[i]==start:
            if toLong==True:
                signal = 'long'
            else:
                signal = 'short'
            holding = int(load*asset / (strike*multi))
            callValue = holding*closeC*multi*(2*toLong-1)
            putValue = holding*closeP*multi*(0-toLong)
            cash = cash - callValue - putValue - (toLong+1)*holding*5
                
            trade_list.append(
                pd.DataFrame([[underly,codeC,codeP,strikeC,strikeP,signal,
                               dfC.index[i],holding,closeC,closeP,callValue,putValue]],
                      columns=['underlying','symbolC','symbolP','strikeC','strikeP','signal',
                               'openTime','holding','closeC','closeP','callValue','putValue']))
        
        if dfC.index[i]==end:
            callClear = holding*closeC*multi*(1-2*toLong)
            putClear = holding*closeP*multi*(toLong-0)
            asset = cash - callClear - putClear - (toLong+1)*holding*5
            
            trade_list.append(
                pd.DataFrame([[dfC.index[i], closeC, closeP,
                               -(callClear + callValue), -(putValue + putClear),
                               -(callClear + callValue + putValue + putClear)]],
                      columns=['clearTime','closeC','closeP','profitC','profitP','profit']))
            break  # 平仓的asset算出来，但是不计入开仓所在的记录表。
        
        asset = cash + holding*(closeC*(2*toLong-1)+closeP*(0-toLong))*multi  # long call, short put
        asset_list.append(asset)

    dfC_ = dfC[dfC.index < end]        
    res = pd.DataFrame(asset_list, columns=['netAsset'], index=dfC_.index)
    resTr = pd.concat(trade_list, axis=1)
        
    return res, asset, resTr


#%%

def run_tradeTime(symbol, cycles=10, cash=100000.0, holding=0, load=1.0, number=1, tradeDF=[]):
    
    asset_list = []
    trade_list = []

    newCash = cash
    newLoad = load
    trDF = tradeDF
    
    for i in range(len(trDF)-1):
        
        openTime = trDF.iloc[i][0]
        clearTime = trDF.iloc[i+1][0]
        signal = trDF.iloc[i][1] > 0  # signal为True，多头信号     
      
        temp = run_everyTrade(df, cash=newCash, holding=0, load=newLoad, 
                              toLong=signal, start=openTime, end=clearTime, multi=10000)
        
        asset_list.append(temp[0])
        newCash = temp[1]
        trade_list.append(temp[2])
    
    res = pd.concat(asset_list)
    resTr = pd.concat(trade_list)
    resTr['group']=[number]*len(resTr)
        
    return res, resTr


#%%
# df为原始数据表，cycles为K线周期，period为移动窗口的长度(开仓信号)

def backtest(symbol, cycles=10, cash=1000000.0, holding=0, load=0.1, trade_df=[[]], expterm='m2', atm=[-1,1]):

    trDF = trade_df
    start = trDF.iloc[0][0]
    end = trDF.iloc[-1][0]

    df_ = get_etf(db_etf, symbol, start, end)
    etf = df_[(cycles-1)::cycles]
    
    asset = []
    trade = []

    cols = trDF.columns
    num = int(len(cols)/2)
    
    for i in range(num):
        tr_ = trDF[cols[(2*i):(2*i+2)]].dropna()
        temp = run_tradeTime(symbol, cycles, cash/num, holding, load, i+1, tr_, expterm, atm)
        asset.append(temp[0])
        trade.append(temp[1])
        
    res = pd.concat(asset, axis=1, join='outer').fillna(cash/num)
    res['sum_asset'] = res.apply(lambda x:x.sum(),axis=1)
    
    netAsset = pd.DataFrame(res['sum_asset'], index=res.index)
    netAsset.columns = ['netAsset']
    
    a = etf['close'] / etf['close'][0]
    b = netAsset['netAsset']/netAsset['netAsset'][0]
    Asset = pd.concat([a,b], axis=1, join='outer').fillna(1.0)
    
    recordAll = pd.concat(trade)
    
    return Asset, recordAll, res


#%%
def plot_return(df):
    
    x = np.array(df.index)
    y_etf = np.array(df['close'])
    y_asset = np.array(df['netAsset'])
    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.plot(x,y_etf,label='沪深300净值线')
    plt.plot(x,y_asset,label='策略净值线')
    plt.xticks(x[::int(len(x)/10)], rotation=30)
    plt.legend()
    

#%%
def get_drawback(df):
    result = []
    for j in range(len(df.columns)):
        tmp = df[df.columns[j]]
        drawback = 0.0
        summit = tmp[0]
        bottom = tmp[0]

        for i in range(1,len(tmp)):
            if tmp[i] < tmp[i-1]:
                if tmp[i] < bottom:
                    bottom = tmp[i]
                    drawback = min(drawback, bottom/summit-1)
            elif tmp[i] > tmp[i-1]:
                if tmp[i] > summit:
                    summit = tmp[i]
                    bottom = tmp[i]            
        result.append(drawback)
    
    return result

#%%
def performance(df,record):

    cycle = (pd.Timestamp(df.index[1])-pd.Timestamp(df.index[0])).seconds
    interval = int(4*3600/cycle)  # 相隔的天数
    df_ = df[::interval]
    df_return = (df_ - df_.shift().fillna(1)) / df_.shift().fillna(1)

    return_for_year = (df_.iloc[-1]-df.iloc[0]).values / (len(df_)/242)
    sigma = df_return.std().values * np.sqrt(242)
    drawback = get_drawback(df_)
    sharpe_ratio = (return_for_year - 0.025)/sigma
    MAR_ratio = -return_for_year/drawback
    
    
    s = record['profit']
    win_number = [None, s[s>0].count()]
    loss_number = [None, s[s<0].count()]
    win_ratio = [None, s[s>0].count() / (s[s>0].count() + s[s<0].count())]
    
    win_average = [None, s[s>0].sum() / s[s>0].count()]
    loss_average = [None, s[s<0].sum() / s[s<0].count()]
    profitOverLoss = [None, - s[s>0].sum() / s[s<0].sum()]
    
    stat = [return_for_year,
            sigma,
            drawback,
            sharpe_ratio,
            MAR_ratio,
            
            win_number,
            loss_number,
            win_ratio,
            win_average,
            loss_average,
            profitOverLoss]
    
    result = pd.DataFrame(stat, columns=['etf_300','策略'],
                          index=['return for year',
                                 'sigma',
                                 'drawback',
                                 'sharpe_ratio',
                                 'MAR_ratio',
                                 
                                 'win_number',
                                 'loss_number',
                                 'win_ratio',
                                 'win_average',
                                 'loss_average',
                                 'profits/loss'])
    
    print(result)

