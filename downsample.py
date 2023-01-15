import pandas as pd
import numpy as np
from sklearn import preprocessing, ensemble, linear_model, pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve
from utils import draw_plot


df = pd.read_csv('creditcard.csv')
X, y = df.drop('Class', axis = 1), df['Class']
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2023)

downsampled_df = pd.concat([df.loc[df['Class'] == 0][:492], df.loc[df['Class'] == 1]], axis = 0)
downsampled_X, downsampled_y = downsampled_df.drop('Class', axis = 1), downsampled_df['Class']


attributes = ['Amount', 'Time']
num_transformer = pipeline.Pipeline(steps = ([('inputer', SimpleImputer(strategy='median')),
                                           ('robust_scaler', RobustScaler())]))
preprocessor = ColumnTransformer([('num', num_transformer, attributes)], remainder = 'passthrough')

pipe = pipeline.Pipeline(steps = ([('preprocess', preprocessor),
                                   ('classifier', LogisticRegression(solver = 'liblinear'))]))

PARAMS = {"classifier__penalty": ['l1', 'l2'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_pipe = GridSearchCV(pipe, param_grid = PARAMS, cv = 5, verbose = 1)
grid_pipe.fit(downsampled_X, downsampled_y)

undersample_y_score = grid_pipe.decision_function(X)
precision, recall, _ = precision_recall_curve(y, undersample_y_score)

draw_plot(precision, recall)
