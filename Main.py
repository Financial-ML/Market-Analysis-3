from Features import *
from Models import *
from BackTest import *
import fxcmpy
import pandas as pd
import datetime as dt
import math
from datetime import timedelta
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
#defult data
Sympol = 'GBP/USD'
Period = ['H1','D1']
start_traning = dt.datetime(2018, 6, 1)
stop_traning = dt.datetime(2019, 5, 30)
Algorithm = ['Algo_1']
Distance = 24
start_testing = dt.datetime(2019, 6, 1)
stop_testing = dt.datetime(2019, 9, 30)
#------------------------------------------------------------------------------
#use defult data
print("Use defult data:")
options = ['Yes','No']
choice_num = let_user_pick(options)

if choice_num==1:
    #Sympol
    print("Please choose sympol:")
    options = ['EUR/USD','GBP/USD','AUD/USD']
    choice_num = let_user_pick(options)
    Sympol = options[choice_num]
    
    #Bar period
    print("Please choose bar period:")
    options = [['m1','H1'],['m5','H1'],['m15','H4'],['m30','H4'],['H1','D1']]
    choice_num = let_user_pick(options)
    Period = options[choice_num]
    
    #Traning period time
    #Start
    date_entry = input('Enter the start date of traning period in YYYY-MM-DD format:')
    year, month, day = map(int, date_entry.split('-'))
    start_traning = dt.datetime(year, month, day)
    
    #End
    date_entry = input('Enter the stop date of traning period in YYYY-MM-DD format:')
    year, month, day = map(int, date_entry.split('-'))
    stop_traning = dt.datetime(year, month, day)
    print(" ")
    
    #ML models
    print("Please choose the Algorithm:")
    options = [['Algo_1'],['Algo_2'],['Algo_3']]
    choice_num = let_user_pick(options)
    Algorithm = options[choice_num]
    
    #Distance Market (wave length)
    Distance = input("Please enter Distance for market:")
    Distance = int(Distance)
    
    #tsting period time
    #Start
    date_entry = input('Enter the start date of testing period in YYYY-MM-DD format:')
    year, month, day = map(int, date_entry.split('-'))
    start_testing = dt.datetime(year, month, day)
    
    #End
    date_entry = input('Enter the stop date of testing period in YYYY-MM-DD format:')
    year, month, day = map(int, date_entry.split('-'))
    stop_testing = dt.datetime(year, month, day)
    print(" ")


#print report
print('---------------------------------')
print("report...                       -")
print('Sympol :',Sympol,'               -')
print('Period :',Period)
print('---------------------------------')
print('Traning period                  -')
print('Start:',start_traning,'     -')
print('End:',stop_traning,'       -')
print('---------------------------------')
print('Algorithm :',Algorithm,'         -')
print('Distance :',Distance)
print('---------------------------------')
print('Testing period                  -')
print('Start:',start_testing,'     -')
print('End:',stop_testing,'       -')
print('---------------------------------')
print(' ')


#conferm input
print("Retrain Models:")
options = ['Yes','No']
choice_num = let_user_pick(options)
#------------------------------------------------------------------------------
#Get Data From Market

#add more bars than 10000
dateTimeDifference = start_traning - stop_traning
dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
dateTimeDifferenceInHours = abs(dateTimeDifferenceInHours)
i = dateTimeDifferenceInHours / 120
i = math.ceil(i)

if choice_num==0:
    TOKEN = "0582abcf967083fa4c627003c7636ff5ee95ebe2"
            
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
            
    if con.is_connected() == True:
        print("Data retrieved...")
        #-----------------------------------------\
        #calculate lot number
        money = con.get_accounts().T.iloc[2]
        money = float(money)
        lot = money / 100
        lot = math.floor(lot)
        #-----------------------------------------\
        for x in range(i-1):
            stop_traning1 = start_traning + timedelta(days=5)
            df = con.get_candles(Sympol, period=Period[0], start=start_traning, stop=stop_traning1)
            df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
            df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
            df = df[['open','high','low','close','volume']]
            df = df[~df.index.duplicated()]
            if x==0:
                prices = df.copy()
            else:
                prices = pd.concat([prices,df])
            start_traning = start_traning + timedelta(days=5)

        df = con.get_candles(Sympol, period=Period[0], start=start_traning, stop=stop_traning)
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

#------------------------------------------------------------------------------
#Process Data
    
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
    marketKey = [0]


    keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
               ,paverageKey,slopeKey,fourierKey,sineKey,marketKey] 


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
    marketDict = Market(prices,marketKey,Distance) 
    print('14') 
    # Create list of dictionaries 
    
    dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
                ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
                ,cciDict.cci,bollingerDict.bands,paverageDict.avs
                ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs,marketDict.slope] 

    # list of column name on csv

    colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
               'cci','bollinger','paverage','slope','fourier','sine','market']

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
    #masterFrameCleaned.to_csv('calculated.csv')
    print('complete procrss the features...')
#------------------------------------------------------------------------------
#Train Models    
    
    Decision_Tree(masterFrameCleaned)
    Random_Forest(masterFrameCleaned)
    KNN(masterFrameCleaned)
    NN(masterFrameCleaned)
    SVM(masterFrameCleaned)
#------------------------------------------------------------------------------
#test 

print("Test the stratgy:")
options = ['Yes','No']
choice_num = let_user_pick(options)
if choice_num==0:
    if Algorithm==['Algo_1']:
        Algo_1(Sympol,Period,start_testing,stop_testing)
    if Algorithm==['Algo_2']:
        Algo_2(Sympol,Period,start_testing,stop_testing)        
    if Algorithm==['Algo_3']:
        Algo_3(Sympol,Period,start_testing,stop_testing)          
#------------------------------------------------------------------------------   
    
    