import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from utils import draw_plot

df = pd.read_csv('creditcard.csv')
X, y = df.drop('Class', axis = 1), df['Class']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2023)

attributes = ['Amount', 'Time']
num_processor = pipeline.Pipeline(steps = ([('imputer', SimpleImputer(strategy = 'median')),
                                            ('robust_scaler', RobustScaler())]))
preprocessing = ColumnTransformer([('num', num_processor, attributes)], remainder = 'passthrough')

PARAMS = {'penalty' : ['l1', 'l2'],
          'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

rand_lr = RandomizedSearchCV(LogisticRegression(solver = 'liblinear'), PARAMS, n_iter=3)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 2023)
smt = SMOTE(sampling_strategy='minority')
imbalanced_pipeline = imbalanced_make_pipeline(preprocessing, smt, rand_lr)

imbalanced_pipeline.fit(Xtrain, ytrain)
best_est = rand_lr.best_estimator_
y_score = best_est.decision_function(preprocessing.transform(Xtest))

precision, recall, _ = precision_recall_curve(ytest, y_score)
draw_plot(precision, recall)