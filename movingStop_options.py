# -*- coding: utf-8 -*-
"""
@author: yuyou
"""

import pandas as pd
import numpy as np
import dolphindb as ddb
import matplotlib.pyplot as plt
import datetime as dt


#%% # download data from dolphindb
s = ddb.session()
s.connect('10.0.60.55',8508)
s.login('reader','888888')

#%%
db_etf = s.loadTable(dbPath = 'dfs://STOCK_SH_TRDMIN', tableName = 'STOCK_SH_TRDMIN')

cols = ['symbol','tradingDay','time','open','close','high','low']
temp = db_etf.where("symbol='510300'").where("date>2019.12.20").where("time!=09:30:00").where("cycle=1").select(cols)
etf_300 = temp.toDF()

#%%  
db_option1 = s.loadTable(dbPath = 'dfs://G2OptionTickStatMinute', tableName = 'TickStat')
cols1 = ['symbol', 'expterm', 'tradingDay', 'time', 'cp', 'last_underly', 'atm_underly']

temp1 = db_option1.where(["underlying='510300'",
                          "tradingDay >= {}".format('2019.12.20'),
                          "expterm='m2'",
                          "atm_underly>-2",
                          "atm_underly<2"]).select(cols1)

etf_options1 = temp1.toDF()

#%%
db_option2 = s.loadTable(dbPath = 'dfs://SHSOL1_TRDMIN_3', tableName = 'SHSOL1_TRDMIN')
cols2 = ['symbol', 'tradingDay', 'time', 'open', 'close']

# temp2 = db_option2.where(["underlying = '510300'",
#                           "tradingDay > 2019.12.20",
#                           "cycle = 1"]).select(cols2)

# etf_options2 = temp2.toDF()


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
        # date_ = datetime.strftime(Timestamp, '%Y-%m-%d')
    df_['time_'] = time_group
    temp = df_.sort_values('time_')
    result = temp.set_index('time_',drop=True)
    
    return result


#%%  # get_tradeTime(df_records, start, end)  # start, end为时间字符串
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

def get_symbol(t, symbols_df):
    
    calls_1 = symbols_df[(symbols_df['cp']=='c') & (symbols_df['atm_underly']==-1)]
    puts__1 = symbols_df[(symbols_df['cp']=='p') & (symbols_df['atm_underly']==1)]
    c = calls_1[calls_1.index > t].symbol[0]  # code of call
    p = puts__1[puts__1.index > t].symbol[0]  # code of put
    
    return c, p

#%%  # 获得特定的期权行情
db_option2 = s.loadTable(dbPath = 'dfs://SHSOL1_TRDMIN_3', tableName = 'SHSOL1_TRDMIN')
cols2 = ['symbol', 'tradingDay', 'time', 'open', 'close']

def get_prices(symbol, start, end, cycles):
    
    c = symbol
    t1 = start
    t2 = end
    cols2 = ['symbol', 'tradingDay', 'time', 'open', 'close']
    cc = db_option2.where(["underlying = '510300'",
                           "symbol = '{}'".format(c),
                           "tradingDay >= {}".format(t1.split()[0]),
                           "tradingDay <= {}".format(t2),
                           "cycle = 1"]).select(cols2)
    
    ccc = add_timeStr(cc.toDF())
    ccc = ccc[(cycles-1)::cycles]
    
    return ccc

#%%  # dfC和dfP为通过symbol和start、end获取的call、put行情表

def run_everyTrade(C_, P_, cash=100000.0, holding=0, load=0.05, toLong=True, start='2019', end='2022'):
    
    code = C_.symbol[0]
    dfC = C_[C_.index>=start]
    dfP = P_.copy()
    asset = cash + holding * (dfC['close'][0]-dfP['close'][0]) * 10000
    asset_list = []
    trade_list = []
    
    callValue = 0.0; putValue = 0.0; callClear = 0.0; putClear = 0.0
    signal = ''
    
    for i in range(len(dfC)):

        closeC = dfC['close'][i]
        closeP = dfP['close'][i]
        
        if dfC.index[i]==start:
            if toLong==True:
                holding = int(load*asset / ((closeC+closeP)*10000)); signal='long'
            else:
                holding = -int(0.3*load*asset / ((closeC+closeP)*10000)); signal='short'
                
            callValue = holding*closeC*10000
            putValue = holding*closeP*10000
            tradeValue = (callValue-putValue)*1.0005
            cash = cash - tradeValue
            trade_list.append(
                pd.DataFrame([[code,signal,dfC.index[i],callValue,-putValue]],
                             columns=['symbol','signal','openTime','callValue','putValue']))
        
        if dfC.index[i]==end:
            callClear = holding*closeC*10000
            putClear = holding*closeP*10000
            clearValue = (callClear-putClear)*0.9995
            asset = cash + clearValue
            trade_list.append(
                pd.DataFrame([[dfC.index[i], callClear, -putClear,
                               callClear-callValue, putValue-putClear,
                               callClear-callValue + putValue-putClear]],
                             columns=['clearTime','callClear','putClear','profitC','profitP','profit']))
            break
        
        asset = cash + holding*(closeC-closeP)*10000  # long call, short put
        asset_list.append(asset)

    dfC_ = dfC[dfC.index < end]        
    res = pd.DataFrame(asset_list, columns=['netAsset'], index=dfC_.index)
    resTr = pd.concat(trade_list, axis=1)
        
    return res, asset, resTr


#%%  # dfC=calls_1, dfP=puts__1, tradeDF=单个组合的交易记录(2列的df)

def run_tradeTime(symbolDF, cycles=10, cash=100000.0, holding=0, load=0.05, number=1, tradeDF=[]):
    
    asset_list = []
    trade_list = []
    newCash = cash
    newLoad = load
    trDF = tradeDF
    
    for i in range(len(trDF)-1):
        
        openTime = trDF.iloc[i][0]
        clearTime = trDF.iloc[i+1][0]
        signal = trDF.iloc[i][1] > 0  # signal为True，多头信号     
        
        C,P = get_symbol(openTime, symbolDF)
        ccc = get_prices(C, openTime, clearTime, cycles)
        ppp = get_prices(P, openTime, clearTime, cycles)
      
        temp = run_everyTrade(ccc, ppp, cash=newCash, holding=0, load=newLoad, toLong=signal,
                              start=openTime, end=clearTime)
        
        asset_list.append(temp[0])
        newCash = temp[1]
        trade_list.append(temp[2])
    
    res = pd.concat(asset_list)
    resTr = pd.concat(trade_list)
    resTr['group']=[number]*len(resTr)
        
    return res, resTr


#%%
# df为现货数据表，cycles为K线周期，period为移动窗口的长度(开仓信号)

def backtest(df, symbolDF, cycles=10, cash=1000000.0, holding=0, load=0.1, trade_df=[[]]):

    trDF = trade_df
    start = trDF.iloc[0][0]
    end = trDF.iloc[-1][0]
    df_ = add_timeStr(df)
    etf = df_[ (df_.index>=start.split()[0]) & (df_.index<end) ][(cycles-1)::cycles]
    
    asset = []
    trade = []

    cols = trDF.columns
    num = int(len(cols)/2)
    
    for i in range(num):
        tr_ = trDF[cols[(2*i):(2*i+2)]].dropna()
        temp = run_tradeTime(symbolDF, cycles, cash/num, holding, load, i+1, tr_)
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


#%%  # plot return
def plot_return(df):
    
    x = np.array(df.index)
    y_etf = np.array(df['close'])
    y_asset = np.array(df['netAsset'])
    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.plot(x,y_etf,label='沪深300收益率')
    plt.plot(x,y_asset,label='移动止损策略收益率')
    plt.xticks(x[::int(len(x)/10)], rotation=30)
    plt.legend()
    
#%%  # get_drawback(df)
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
    interval = int(4*3600/cycle)
    
    df_return = df.iloc[-1] - df.iloc[0]
    return_for_year = df_return.values / (len(df)/interval /242)
    sigma = df.std().values / np.sqrt(len(df)/interval /242)
    drawback = get_drawback(df)
    sharpe_ratio = (return_for_year - 0.015)/sigma
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
    
    result = pd.DataFrame(stat, columns=['etf_300','合成期货策略'],
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
    
#%%
path = 'C:/Users/yuyou/Desktop/test/'
trade_df = pd.read_excel(path+'tradeRecords.xlsx',sheet_name = '交易记录')
records = get_tradeTime(trade_df, '2019.12.23', '2022')

#%%
symbols = add_timeStr(etf_options1)

#%%
aaa = backtest(etf_300, symbols, cycles=10, cash=1000000.0, holding=0, load=0.1, trade_df=records)
performance(aaa[0],aaa[1])
plot_return(aaa[0])




