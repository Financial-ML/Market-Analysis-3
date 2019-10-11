import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

#------------------------------------------------------------------------------
class Holder:
    1
#------------------------------------------------------------------------------

def Decision_Tree(df):
    exists = os.path.isfile('saved_model_dt.pkl')
    if exists:
        os.remove('saved_model_dt.pkl')

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
               ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
               ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']

    labels = df['market0Market'].values
    features = df[list(columns)].values

    Standarization = False

    if Standarization == True:
        minmax = False
        Standard = True
        if minmax == True:
            min_max = MinMaxScaler()
            newfeatures = min_max.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
        elif Standard == True:
            std = StandardScaler()
            newfeatures = std.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)


    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    print ('DT Traning Data accuracy ', accuracy*100)

    accuracy = clf.score(X_test, y_test)
    print ('DT Testing Data accuracy ', accuracy*100)
    
    print('------------------------------------------')
    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model_dt.pkl')
    

#------------------------------------------------------------------------------
    
def Random_Forest(df):
    exists = os.path.isfile('saved_model_rf.pkl')
    if exists:
        os.remove('saved_model_rf.pkl')

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
               ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
               ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']
    
    labels = df['market0Market'].values
    features = df[list(columns)].values

    Standarization = False

    if Standarization == True:
        minmax = False
        Standard = True
        if minmax == True:
            min_max = MinMaxScaler()
            newfeatures = min_max.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
        elif Standard == True:
            std = StandardScaler()
            newfeatures = std.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    print ('RF Traning Data accuracy ', accuracy*100)

    accuracy = clf.score(X_test, y_test)
    print ('RF Testing Data accuracy ', accuracy*100)

    print('------------------------------------------')

    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model_rf.pkl')

#------------------------------------------------------------------------------
    
def KNN(df):
    exists = os.path.isfile('saved_model_knn.pkl')
    if exists:
        os.remove('saved_model_knn.pkl')

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
               ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
               ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']
    
    labels = df['market0Market'].values
    features = df[list(columns)].values

    Standarization = True

    if Standarization == True:
        minmax = False
        Standard = True
        if minmax == True:
            min_max = MinMaxScaler()
            newfeatures = min_max.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
        elif Standard == True:
            std = StandardScaler()
            newfeatures = std.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    clf = KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    print ('KNN Traning Data accuracy ', accuracy*100)

    accuracy = clf.score(X_test, y_test)
    print ('KNN Testing Data accuracy ', accuracy*100)

    print('------------------------------------------')

    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model_knn.pkl')

#------------------------------------------------------------------------------
    
def NN(df):
    exists = os.path.isfile('saved_model_nn.pkl')
    if exists:
        os.remove('saved_model_nn.pkl')

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
               ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
               ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']
    
    labels = df['market0Market'].values
    features = df[list(columns)].values

    Standarization = True

    if Standarization == True:
        minmax = False
        Standard = True
        if minmax == True:
            min_max = MinMaxScaler()
            newfeatures = min_max.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
        elif Standard == True:
            std = StandardScaler()
            newfeatures = std.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    clf = MLPClassifier(activation='relu',max_iter=100000)
    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    print ('NN Traning Data accuracy ', accuracy*100)

    accuracy = clf.score(X_test, y_test)
    print ('NN Testing Data accuracy ', accuracy*100)

    print('------------------------------------------')

    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model_nn.pkl')
    
#------------------------------------------------------------------------------
    
def SVM(df):
    exists = os.path.isfile('saved_model_svm.pkl')
    if exists:
        os.remove('saved_model_svm.pkl')

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
               ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
               ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']
    
    labels = df['market0Market'].values
    features = df[list(columns)].values

    Standarization = True

    if Standarization == True:
        minmax = False
        Standard = True
        if minmax == True:
            min_max = MinMaxScaler()
            newfeatures = min_max.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
        elif Standard == True:
            std = StandardScaler()
            newfeatures = std.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(newfeatures, labels, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    clf = svm.SVC(kernel='linear')
    clf = clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    print ('SVM Traning Data accuracy ', accuracy*100)

    accuracy = clf.score(X_test, y_test)
    print ('SVM Testing Data accuracy ', accuracy*100)

    print('------------------------------------------')

    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model_svm.pkl')
    
    
    
    