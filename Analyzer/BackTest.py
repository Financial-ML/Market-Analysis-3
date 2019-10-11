import fxcmpy
import pandas as pd
import math
import numpy as np
from datetime import timedelta
from Features import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------
class Holder:
    1
#------------------------------------------------------------------------------

def let_user_pick(options):
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1,element))
    i = input("Enter number: ")
    print(" ")
    try:
        if 0 < int(i) <= len(options):
            return int(i)-1
    except:
        pass
    return None
#------------------------------------------------------------------------------
def Convert(string): 
    li = list(string.split(" ")) 
    return li 
#------------------------------------------------------------------------------
'''
functions
1- open buy
2- open sell
3- close any
'''
#---------------------------- 
state = []
c = 0
def open_buy(price):
    global state
    state = [1 , price]
#----------------------------  
def open_sell(price):
    global state
    state = [0 , price]
#----------------------------  
def close_all(price):
    profit = 0 
    global state
    global c
    if len(state)!=0:
        if state[0]==0:
            profit = state[1] - price - 0.0002
        elif state[0]==1:
            profit = price - state[1] - 0.0002
        c = c + 1
    state = []
    
    return profit
#------------------------------------------------------------------------------

def Algo_1(Sympol,Period,start_testing,stop_testing):
#------------------------------------------------------------------------------
    #defult data
    Models = ['DT','RF','KNN','NN']
    NTime = 25
    Action = ['Buy','Sell']
#------------------------------------------------------------------------------    
    #use defult data
    print("Use defult data:")
    options = ['Yes','No']
    choice_num = let_user_pick(options)

    if choice_num==1:
        #Models
        Models = input("Please choose ML-Models in this format DT RF KNN NN SVM :")
        Models = Convert(Models)
        
        #NTime
        NTime = input("Please enter NTime (predict every N Time):")
        NTime = int(NTime)
        
        #Action
        print("Please choose the Action:")
        options = [['Buy','Sell'], ['Buy','Hold'], ['Sell','Hold']]
        choice_num = let_user_pick(options)
        Action = options[choice_num]
        print(" ")
        
    #print report
    print('---------------------------------')
    print("Algo report...")
    print('Models :',Models)
    print('NTime :',NTime)
    print('Action :',Action)
    print('---------------------------------')
    print(' ')    


    #conferm input
    print("Conferm input:")
    options = ['Yes','No']
    choice_num = let_user_pick(options)
    
#------------------------------------------------------------------------------
    #add more bars than 10000
    dateTimeDifference = start_testing - stop_testing
    dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
    dateTimeDifferenceInHours = abs(dateTimeDifferenceInHours)
    i = dateTimeDifferenceInHours / 120
    i = math.ceil(i)
    
    if choice_num==0:
        TOKEN = "0582abcf967083fa4c627003c7636ff5ee95ebe2"
                
        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                
        if con.is_connected() == True:
            print("Data retrieved...")
            
            for x in range(i-1):
                stop_testing1 = start_testing + timedelta(days=5)
                df = con.get_candles(Sympol, period=Period[0], start=start_testing, stop=stop_testing1)
                df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
                df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
                df = df[['open','high','low','close','volume']]
                df = df[~df.index.duplicated()]
                if x==0:
                    prices = df.copy()
                else:
                    prices = pd.concat([prices,df])
                start_testing = start_testing + timedelta(days=5)
    
            df = con.get_candles(Sympol, period=Period[0], start=start_testing, stop=stop_testing)
            df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow']) 
            df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
            df = df[['open','high','low','close','volume']]
            df = df[~df.index.duplicated()]
            if i==1:
                prices = df.copy() 
            else:
                prices = pd.concat([prices,df])
        else:
            print('No connection with fxcm')
        prices = prices.drop_duplicates(keep=False)
        
        prices['date']=pd.to_datetime(prices.index)
        
        if Period[1]=='H1':
            for i in range(0,len(prices)):
                if prices.date.dt.minute.iloc[i] == 0:
                    break
            prices = prices[i:]
        
        if Period[1]=='D1' or Period[1]=='H4':
            for i in range(0,len(prices)):
                if prices.date.dt.hour.iloc[i] == 0:
                    break
            prices = prices[i:]
        prices = prices.drop(['date'], axis=1)
        #prices.to_csv('prices.csv')
        
        print('Data is ready to process...')

   
        prices = resamble(prices,Period[1])
        
        momentumKey = [3,4,5,8,9,10] 
        stochasticKey = [3,4,5,8,9,10] 
        williamsKey = [6,7,8,9,10] 
        procKey = [12,13,14,15] 
        wadlKey = [15] 
        adoscKey = [2,3,4,5] 
        macdKey = [15,30] 
        cciKey = [15] 
        bollingerKey = [15] 
        paverageKey = [2] 
        slopeKey = [3,4,5,10,20,30] 
        fourierKey = [10,20,30] 
        sineKey = [5,6] 
    
    
    
        keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
                   ,paverageKey,slopeKey,fourierKey,sineKey] 
    
    
        momentumDict = momentum(prices,momentumKey) 
        print('1') 
        stochasticDict = stochastic(prices,stochasticKey) 
        print('2') 
        williamsDict = williams(prices,williamsKey)
        print('3') 
        procDict = proc(prices,procKey) 
        print('4') 
        wadlDict = wadl(prices,wadlKey) 
        print('5')
        adoscDict = adosc(prices,adoscKey)
        print('6') 
        macdDict = macd(prices,macdKey) 
        print('7') 
        cciDict = cci(prices,cciKey) 
        print('8')
        bollingerDict = bollinger(prices,bollingerKey,2) 
        print('9') 
        paverageDict = pavarage(prices,paverageKey) 
        print('10') 
        slopeDict = slopes(prices,slopeKey) 
        print('11') 
        fourierDict = fourier(prices,fourierKey) 
        print('12') 
        sineDict = sine(prices,sineKey) 
        print('13') 
    
        # Create list of dictionaries 
        
        dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
                    ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
                    ,cciDict.cci,bollingerDict.bands,paverageDict.avs
                    ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs] 
    
        # list of column name on csv
    
        colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
                   'cci','bollinger','paverage','slope','fourier','sine']
    
        masterFrame = pd.DataFrame(index = prices.index) 
        for i in range(0,len(dictlist)): 
            if colFeat[i] == 'macd':
                colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1]) 
                masterFrame[colID] = dictlist[i] 
            else: 
                for j in keylist[i]: 
                    for k in list(dictlist[i][j]):
                        colID = colFeat[i] + str(j) + str(k)
                        masterFrame[colID] = dictlist[i][j][k]
                            
        threshold = round(0.7*len(masterFrame)) 
        masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]
    
        masterFrameCleaned = masterFrame.copy() 
        masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
        masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
        masterFrameCleaned['date']=pd.to_datetime(masterFrameCleaned.index)
        
        if Period[1]=='H1':
            for i in range(0,len(masterFrameCleaned)):
                if masterFrameCleaned.date.dt.minute.iloc[i] == 0:
                    break
            masterFrameCleaned = masterFrameCleaned[i:]
        
        if Period[1]=='D1' or Period[1]=='H4':
            for i in range(0,len(masterFrameCleaned)):
                if masterFrameCleaned.date.dt.hour.iloc[i] == 0:
                    break
            masterFrameCleaned = masterFrameCleaned[i:]
        masterFrameCleaned = masterFrameCleaned.drop(['date'], axis=1)
        
        #masterFrameCleaned.to_csv('calculated.csv')
        print('complete procrss the features...')
#------------------------------------------------------------------------------
        #load the models
        columns = ['momentum3close','momentum4close'
                   ,'momentum5close','momentum8close','momentum9close','momentum10close'
                   ,'stoch3K','stoch3D','stoch4K','stoch4D'
                   ,'stoch5K','stoch5D','stoch8K','stoch8D'
                   ,'stoch9K','stoch9D','stoch10K'
                   ,'stoch10D','will6R','will7R','will8R'
                   ,'will9R','will10R','proc12close','proc13close'
                   ,'proc14close','proc15close','wadl15close','adosc2AD'
                   ,'adosc3AD','adosc4AD','adosc5AD','macd1530','cci15close'
                   ,'bollinger15upper','bollinger15mid','bollinger15lower','paverage2open'
                   ,'paverage2high','paverage2low','paverage2close','slope3high','slope4high','slope5high'
                   ,'slope10high','slope20high','slope30high'
                   ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0'
                   ,'fourier20a1','fourier20b1','fourier20w','fourier30a0'
                   ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w'
                   ,'sine6a0','sine6b1','sine6w','open','high','low','close']


        clf_load_dt = joblib.load('saved_model_dt.pkl')
        clf_load_knn = joblib.load('saved_model_knn.pkl')
        clf_load_nn = joblib.load('saved_model_nn.pkl')
        clf_load_rf = joblib.load('saved_model_rf.pkl')
        clf_load_svm = joblib.load('saved_model_svm.pkl')
        
        df = masterFrameCleaned[list(columns)].values
        std = StandardScaler()
        newfeaturess = std.fit_transform(df)
        
        a=[]
        profit = 0
        sum_profit = 0
        l = 0
        w = 0
        maxx = 0
        minn = 0
        
        for i in range(0,len(masterFrameCleaned)):
            profit = 0
            if i % NTime == 0:
                a=[]
                newfeatures = newfeaturess[i].reshape(1, 69) 
                dff = df[i].reshape(1, 69)
                if 'DT' in Models:
                    predicted_dt = clf_load_dt.predict(dff)
                    a = np.append(a,predicted_dt)
                if 'RF' in Models:
                    predicted_rf = clf_load_rf.predict(dff)
                    a = np.append(a,predicted_rf)
                if 'KNN' in Models:
                    predicted_knn = clf_load_knn.predict(newfeatures)
                    a = np.append(a,predicted_knn)
                if 'NN' in Models:
                    predicted_nn = clf_load_nn.predict(newfeatures)
                    a = np.append(a,predicted_nn)
                if 'SVM' in Models:
                    predicted_svm = clf_load_svm.predict(newfeatures)
                    a = np.append(a,predicted_svm)
#------------------------------------------------------------------------------   
#doing actions                    
                nTemp = a[0]
                bEqual = True
 
                for item in a:
                    if nTemp != item:
                        bEqual = False
                        break;
#----------------------------                
                if Action ==['Buy','Sell']:
                    if bEqual:
                        if a[0]==0:
                            #print("open sell close buy")
                            if len(state)!=0:
                                if state[0]==1:
                                    profit = close_all(masterFrameCleaned.close.iloc[i])
                                else:
                                    continue
                            open_sell(masterFrameCleaned.close.iloc[i])
                        elif a[0]==1:
                            #print("open buy close sell")
                            if len(state)!=0:
                                if state[0]==0:
                                    profit = close_all(masterFrameCleaned.close.iloc[i])
                                else: 
                                    continue
                            open_buy(masterFrameCleaned.close.iloc[i])                            
                    else:
                        #print("Close anything")
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#----------------------------                        
                if Action ==['Buy','Hold']:
                    if bEqual:
                        if a[0]==1:
                            #print("open buy")
                            if len(state)==0:
                                open_buy(masterFrameCleaned.close.iloc[i])
                            else:
                                continue
                        else:
                            #print("Close anything")
                            profit = close_all(masterFrameCleaned.close.iloc[i])
                    else:
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#----------------------------                        
                if Action ==['Sell','Hold']:
                    if bEqual:
                        if a[0]==0:
                            #print("open sell")
                            if len(state)==0:
                                open_sell(masterFrameCleaned.close.iloc[i])
                            else:
                                continue
                        else:
                            #print("Close anything")
                            profit = close_all(masterFrameCleaned.close.iloc[i])
                    else:
                        #print("Close anything")
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#----------------------------
                if profit < 0:
                    l = l + 1
                if profit > 0:
                    w = w + 1    
                if maxx < profit: 
                    maxx = profit
                if minn > profit: 
                    minn = profit 
                sum_profit = sum_profit + profit
        print('profit of 10 lots:',sum_profit*10000)
        print('Total number of trades:',c)
        print('sum of wining trades:',w)
        print('sum of loss trades:',l)
        print('max Down for 10 lots:',minn*10000)
        print('max up for 10 lots:',maxx*10000)
#------------------------------------------------------------------------------

def Algo_2(Sympol,Period,start_testing,stop_testing):
#------------------------------------------------------------------------------
    #defult data
    Models = ['DT','RF','KNN','NN']
    NTime = 25 
    Action = ['Buy','Sell']
#------------------------------------------------------------------------------    
    #use defult data
    print("Use defult data:")
    options = ['Yes','No']
    choice_num = let_user_pick(options)

    if choice_num==1:
        #Models
        Models = input("Please choose ML-Models in this format DT RF KNN NN SVM :")
        Models = Convert(Models)
        
        #NTime
        NTime = input("Please enter NTime (predict every N Time):")
        NTime = int(NTime)
        
        #Action
        print("Please choose the Action:")
        options = [['Buy','Sell'], ['Buy','Hold'], ['Sell','Hold']]
        choice_num = let_user_pick(options)
        Action = options[choice_num]
        print(" ")
        
    #print report
    print('---------------------------------')
    print("Algo report...")
    print('Models :',Models)
    print('NTime :',NTime)
    print('Action :',Action)
    print('---------------------------------')
    print(' ')    


    #conferm input
    print("Conferm input:")
    options = ['Yes','No']
    choice_num = let_user_pick(options)
    
#------------------------------------------------------------------------------
    #add more bars than 10000
    dateTimeDifference = start_testing - stop_testing
    dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
    dateTimeDifferenceInHours = abs(dateTimeDifferenceInHours)
    i = dateTimeDifferenceInHours / 120
    i = math.ceil(i)
    
    if choice_num==0:
        TOKEN = "0582abcf967083fa4c627003c7636ff5ee95ebe2"
                
        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                
        if con.is_connected() == True:
            print("Data retrieved...")
            
            for x in range(i-1):
                stop_testing1 = start_testing + timedelta(days=5)
                df = con.get_candles(Sympol, period=Period[0], start=start_testing, stop=stop_testing1)
                df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
                df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
                df = df[['open','high','low','close','volume']]
                df = df[~df.index.duplicated()]
                if x==0:
                    prices = df.copy()
                else:
                    prices = pd.concat([prices,df])
                start_testing = start_testing + timedelta(days=5)
    
            df = con.get_candles(Sympol, period=Period[0], start=start_testing, stop=stop_testing)
            df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow']) 
            df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
            df = df[['open','high','low','close','volume']]
            df = df[~df.index.duplicated()]
            if i==1:
                prices = df.copy() 
            else:
                prices = pd.concat([prices,df])
        else:
            print('No connection with fxcm')
        prices = prices.drop_duplicates(keep=False)
        
        prices['date']=pd.to_datetime(prices.index)
        
        if Period[1]=='H1':
            for i in range(0,len(prices)):
                if prices.date.dt.minute.iloc[i] == 0:
                    break
            prices = prices[i:]
        
        if Period[1]=='D1' or Period[1]=='H4':
            for i in range(0,len(prices)):
                if prices.date.dt.hour.iloc[i] == 0:
                    break
            prices = prices[i:]
        prices = prices.drop(['date'], axis=1)
        #prices.to_csv('prices.csv')
        
        print('Data is ready to process...')

   
        prices = resamble(prices,Period[1])
        
        momentumKey = [3,4,5,8,9,10] 
        stochasticKey = [3,4,5,8,9,10] 
        williamsKey = [6,7,8,9,10] 
        procKey = [12,13,14,15] 
        wadlKey = [15] 
        adoscKey = [2,3,4,5] 
        macdKey = [15,30] 
        cciKey = [15] 
        bollingerKey = [15] 
        paverageKey = [2] 
        slopeKey = [3,4,5,10,20,30] 
        fourierKey = [10,20,30] 
        sineKey = [5,6] 
    
    
    
        keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
                   ,paverageKey,slopeKey,fourierKey,sineKey] 
    
    
        momentumDict = momentum(prices,momentumKey) 
        print('1') 
        stochasticDict = stochastic(prices,stochasticKey) 
        print('2') 
        williamsDict = williams(prices,williamsKey)
        print('3') 
        procDict = proc(prices,procKey) 
        print('4') 
        wadlDict = wadl(prices,wadlKey) 
        print('5')
        adoscDict = adosc(prices,adoscKey)
        print('6') 
        macdDict = macd(prices,macdKey) 
        print('7') 
        cciDict = cci(prices,cciKey) 
        print('8')
        bollingerDict = bollinger(prices,bollingerKey,2) 
        print('9') 
        paverageDict = pavarage(prices,paverageKey) 
        print('10') 
        slopeDict = slopes(prices,slopeKey) 
        print('11') 
        fourierDict = fourier(prices,fourierKey) 
        print('12') 
        sineDict = sine(prices,sineKey) 
        print('13') 
    
        # Create list of dictionaries 
        
        dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
                    ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
                    ,cciDict.cci,bollingerDict.bands,paverageDict.avs
                    ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs] 
    
        # list of column name on csv
    
        colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
                   'cci','bollinger','paverage','slope','fourier','sine']
    
        masterFrame = pd.DataFrame(index = prices.index) 
        for i in range(0,len(dictlist)): 
            if colFeat[i] == 'macd':
                colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1]) 
                masterFrame[colID] = dictlist[i] 
            else: 
                for j in keylist[i]: 
                    for k in list(dictlist[i][j]):
                        colID = colFeat[i] + str(j) + str(k)
                        masterFrame[colID] = dictlist[i][j][k]
                            
        threshold = round(0.7*len(masterFrame)) 
        masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]
    
        masterFrameCleaned = masterFrame.copy() 
        masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
        masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
        masterFrameCleaned['date']=pd.to_datetime(masterFrameCleaned.index)
        
        if Period[1]=='H1':
            for i in range(0,len(masterFrameCleaned)):
                if masterFrameCleaned.date.dt.minute.iloc[i] == 0:
                    break
            masterFrameCleaned = masterFrameCleaned[i:]
        
        if Period[1]=='D1' or Period[1]=='H4':
            for i in range(0,len(masterFrameCleaned)):
                if masterFrameCleaned.date.dt.hour.iloc[i] == 0:
                    break
            masterFrameCleaned = masterFrameCleaned[i:]
        masterFrameCleaned = masterFrameCleaned.drop(['date'], axis=1)
        
        #masterFrameCleaned.to_csv('calculated.csv')
        print('complete procrss the features...')
#------------------------------------------------------------------------------
        #load the models
        columns = ['momentum3close','momentum4close'
                   ,'momentum5close','momentum8close','momentum9close','momentum10close'
                   ,'stoch3K','stoch3D','stoch4K','stoch4D'
                   ,'stoch5K','stoch5D','stoch8K','stoch8D'
                   ,'stoch9K','stoch9D','stoch10K'
                   ,'stoch10D','will6R','will7R','will8R'
                   ,'will9R','will10R','proc12close','proc13close'
                   ,'proc14close','proc15close','wadl15close','adosc2AD'
                   ,'adosc3AD','adosc4AD','adosc5AD','macd1530','cci15close'
                   ,'bollinger15upper','bollinger15mid','bollinger15lower','paverage2open'
                   ,'paverage2high','paverage2low','paverage2close','slope3high','slope4high','slope5high'
                   ,'slope10high','slope20high','slope30high'
                   ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0'
                   ,'fourier20a1','fourier20b1','fourier20w','fourier30a0'
                   ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w'
                   ,'sine6a0','sine6b1','sine6w','open','high','low','close']


        clf_load_dt = joblib.load('saved_model_dt.pkl')
        clf_load_knn = joblib.load('saved_model_knn.pkl')
        clf_load_nn = joblib.load('saved_model_nn.pkl')
        clf_load_rf = joblib.load('saved_model_rf.pkl')
        clf_load_svm = joblib.load('saved_model_svm.pkl')
        
        df = masterFrameCleaned[list(columns)].values
        std = StandardScaler()
        newfeaturess = std.fit_transform(df)
        
        a=[]
        profit = 0
        sum_profit = 0
        l = 0
        w = 0
        maxx = 0
        minn = 0
        one = 0
        two = 0
        for i in range(0,len(masterFrameCleaned)):
            profit = 0
            if i % (NTime+2) == 0:
                one = 0
                two = 0
            if i % NTime == 0 or i % (NTime+1) ==0:
                a=[]
                newfeatures = newfeaturess[i].reshape(1, 69) 
                dff = df[i].reshape(1, 69)
                if 'DT' in Models:
                    predicted_dt = clf_load_dt.predict(dff)
                    a = np.append(a,predicted_dt)
                if 'RF' in Models:
                    predicted_rf = clf_load_rf.predict(dff)
                    a = np.append(a,predicted_rf)
                if 'KNN' in Models:
                    predicted_knn = clf_load_knn.predict(newfeatures)
                    a = np.append(a,predicted_knn)
                if 'NN' in Models:
                    predicted_nn = clf_load_nn.predict(newfeatures)
                    a = np.append(a,predicted_nn)
                if 'SVM' in Models:
                    predicted_svm = clf_load_svm.predict(newfeatures)
                    a = np.append(a,predicted_svm)
#------------------------------------------------------------------------------   
#doing actions                    
                nTemp = a[0]
                bEqual = True
 
                for item in a:
                    if nTemp != item:
                        bEqual = False
                        break;
#---------------------------- 
                if i % NTime == 0:
                    if bEqual:
                        one = 1
                if i % (NTime+1) == 0:
                    if bEqual:
                        two = 1
#----------------------------                         
                if Action ==['Buy','Sell']:
                    if bEqual:
                        if a[0]==0:
                            #print("open sell close buy")
                            if len(state)!=0:
                                if state[0]==1:
                                    profit = close_all(masterFrameCleaned.close.iloc[i])
                                else:
                                    continue
                            if two==1 and one==1:
                                open_sell(masterFrameCleaned.close.iloc[i])
                        elif a[0]==1:
                            #print("open buy close sell")
                            if len(state)!=0:
                                if state[0]==0:
                                    profit = close_all(masterFrameCleaned.close.iloc[i])
                                else: 
                                    continue
                            if two==1 and one ==1:
                                open_buy(masterFrameCleaned.close.iloc[i])                            
                    else:
                        #print("Close anything")
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#---------------------------- 
#still in bilding                        
                if Action ==['Buy','Hold']:
                    if bEqual:
                        if a[0]==1:
                            #print("open buy")
                            if len(state)==0:
                                if two==1 and one==1:
                                    open_buy(masterFrameCleaned.close.iloc[i])
                            else:
                                continue
                        else:
                            #print("Close anything")
                            profit = close_all(masterFrameCleaned.close.iloc[i])
                    else:
                        #print("Close anything")
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#----------------------------  
#still in bilding    
                if Action ==['Sell','Hold']:
                    if bEqual:
                        if a[0]==0:
                            #print("open sell")
                            if len(state)==0:
                                if two==1 and one==1:
                                    open_sell(masterFrameCleaned.close.iloc[i])
                            else:
                                continue
                        else:
                            #print("Close anything")
                            profit = close_all(masterFrameCleaned.close.iloc[i])
                    else:
                        #print("Close anything")
                        continue
                        profit = close_all(masterFrameCleaned.close.iloc[i])
#----------------------------
                if profit < 0:
                    l = l + 1
                if profit > 0:
                    w = w + 1    
                if maxx < profit: 
                    maxx = profit
                if minn > profit: 
                    minn = profit 
                sum_profit = sum_profit + profit
        print('profit of 10 lots:',sum_profit*10000)
        print('Total number of trades:',c)
        print('sum of wining trades:',w)
        print('sum of loss trades:',l)
        print('max Down for 10 lots:',minn*10000)
        print('max up for 10 lots:',maxx*10000)
#------------------------------------------------------------------------------

def Algo_3(Sympol,Period,start_testing,stop_testing):
    print('Algo_3 is not Ready')
