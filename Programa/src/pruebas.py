
 
from data_preprocessing import *
textos = []
y = []
textos, y = load_texts('../data/despoblacion', 'despoblacion')
textos, y = load_texts('../data/no_despoblacion', 'no_despoblacion', textos, y)
textos, y = load_texts('../data/unlabeled/unlab_despoblacion', 'despoblacion', textos, y)
textos, y = load_texts('../data/unlabeled/unlab_noDespoblacion', 'no_despoblacion', textos, y)

processed_texts = token_stop_stem_lower(textos)
X=cv_ocurrences(processed_texts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




#DECISION TREE----------------------------------------------------------------------------------------------------

from sklearn import tree

DecisionTree = tree.DecisionTreeClassifier(random_state=0)
accCV = crossValidationTest(DecisionTree, 'DECISION TREE', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(DecisionTree, 'DECISION TREE', X_train, y_train, X_test, y_test)
saveResults('decision tree', accCV, accTrain, accTest)




#NAIVE BAYES------------------------------------------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
accCV = crossValidationTest(gnb, 'NAIVE BAYES', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(gnb, 'NAIVE BAYES', X_train, y_train, X_test, y_test)
saveResults('naive bayes', accCV, accTrain, accTest)




#RANDOM FOREST----------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

RndForest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
accCV = crossValidationTest(RndForest, 'RANDOM FOREST', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(RndForest, 'RANDOM FOREST', X_train, y_train, X_test, y_test)
saveResults('random forest 200 trees', accCV, accTrain, accTest)




#KNN--------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
accCV = crossValidationTest(neigh, 'K NEIGHBORS', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(neigh, 'K NEIGHBORS', X_train, y_train, X_test, y_test)
saveResults('k neighbors 5 neigh', accCV, accTrain, accTest)




#GRADIENT BOOSTING------------------------------------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier

grBoost = GradientBoostingClassifier(n_estimators=200, random_state=0)
accCV = crossValidationTest(grBoost, 'GRADIENT BOOSTING', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(grBoost, 'GRADIENT BOOSTING', X_train, y_train, X_test, y_test)
saveResults('gradient boosting 200 est', accCV, accTrain, accTest)




#MULTI-LAYER PERCEPTRON-------------------------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
dnn = MLPClassifier(random_state=0)
#dnn = MLPClassifier(random_state=0, max_iter=100, hidden_layer_sizes=(20,20,20), learning_rate='adaptive')
accCV = crossValidationTest(dnn, 'MULTI-LAYER PERCEPTRON', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(dnn, 'MULTI-LAYER PERCEPTRON', X_train, y_train, X_test, y_test)
saveResults('ML perceptron default', accCV, accTrain, accTest)



#LINEAR SUPPORT VECTOR---------------------------------------------------------------------------------------------
from sklearn.svm import LinearSVC

lsvc =LinearSVC(random_state=0)
accCV = crossValidationTest(lsvc, 'LINEAR SUPPORT VECTOR', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(lsvc, 'LINEAR SUPPORT VECTOR', X_train, y_train, X_test, y_test)
saveResults('linear support vector', accCV, accTrain, accTest)




#SUPPORT VECTOR MACHINE--------------------------------------------------------------------------------------------

from sklearn.svm import SVC

svmModel = SVC(random_state=0)
accCV = crossValidationTest(svmModel, 'SUPPORT VECTOR MACHINE', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(svmModel, 'SUPPORT VECTOR MACHINE', X_train, y_train, X_test, y_test)
saveResults('support vector machine', accCV, accTrain, accTest)




#SGD---------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier(random_state=0, n_jobs=-1)
accCV = crossValidationTest(sgdc, 'Stochastic Gradient Descent', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(sgdc, 'Stochastic Gradient Descent', X_train, y_train, X_test, y_test)
saveResults('Stochastic Gradient Descent', accCV, accTrain, accTest)



#CATBOOST----------------------------------------------------------------------------------------------------------

from catboost import CatBoostClassifier

catB = CatBoostClassifier()
accCV = crossValidationTest(catB, 'CATBOOST', X_train, y_train, folds=10)
accTrain, accTest = predictTrainAndTest(catB, 'CATBOOST', X_train, y_train, X_test, y_test)
saveResults('catBoost', accCV, accTrain, accTest)


