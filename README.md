# Market-Analysis-2

### Overview

In Market Analysis 3 we change its component to improve performance and speed, plus that we merge all the component in two main programes Analyzer and Applier.

# Analyzer Components

* [Dataset](#Dataset)

* [Feature-Calculator](#Feature-Calculator)

* [ML-Models](#ML-Models)

* [Tester](#Tester)


## Dataset

Pull the Dataset for any symbol in any period of time in Forex market that contain the basic Features
(open, high, low, close).

#### How we do it
We do it by using fxcm python library (PYFXCM).

   
## Feature-Calculator

In Feature-Calculator we calculate the machine learning Feature based on this research [(here)](http://www.wseas.us/e-library/conferences/2011/Penang/ACRE/ACRE-05.pdf).

#### How we do it
* We build it in python .
* It based on the main Features [Dataset](#Dataset) and some functions in this research [(here)](http://www.wseas.us/e-library/conferences/2011/Penang/ACRE/ACRE-05.pdf)..


## ML-Models

Different Machine Learning models that we used to learn from the [Feature](#Feature-Calculator).

#### How we do it
* We build the the models in python using scikit-learn.
* It learn from our predefined [Feature](#Feature-Calculator).
* And then save it after finsh traning in PKL file.

#### Models
* Decision Tree.
* k-nearest neighbor.
* RandomForest.
* Support vector machine.
* Neural-network-MLPClassifier.

## Testing

Tool that use to test the strategy that has been developed outside MQL4 in python.

#### How we do it
* We do it by build our algorithmic trading strategy in python.
* And load our Models in Tester program.
* Then calculate the profits by saving the enter price then subtract from it the close price.

#### Algorithmic Trading
We call the models (Decision Tree and RandomForest) every one hour and say if the first 10 min is the same prediction do the predicted action.
#### What we test
1. Profit
1. Total number of trades
1. Sum of wining trades
1. Sum of loss trades
1. Max drawdown
1. Best trade

# Applier Components

