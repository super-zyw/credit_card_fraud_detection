import pandas as pd
import numpy as np
from sklearn import preprocessing, ensemble, linear_model, pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv('creditcard.csv')

print('is_fraud : {:.4f}'.format(sum(df['Class'] == 1) / len(df['Class'])))
print('is_not_fraud : {:.4f}'.format(sum(df['Class'] == 0) / len(df['Class'])))

fig, axis = plt.subplots(1, 3)
df['Time'].plot.hist(ax = axis[0])
df['Amount'].plot.hist(ax = axis[1])
df['Class'].plot.hist(ax = axis[2])
axis[0].set_title('Time')
axis[1].set_title('Amount')
axis[2].set_title('Class')
plt.show()

attributes = ['Amount', 'Time']
num_pipeline = pipeline.Pipeline(steps = ([('robust_scaler', RobustScaler())]))
preprocessing = ColumnTransformer([('num', num_pipeline, attributes)], remainder = 'passthrough')

df = pd.DataFrame(preprocessing.fit_transform(df), columns = df.columns, index = df.index)

df = df.sample(frac = 1, random_state = 2023)
non_fraud = df.loc[df['Class'] == 0][:492]
fraud = df.loc[df['Class'] == 1]

ori_X, ori_y = df.drop('Class', axis = 1), df['Class']

ori_Xtrain, ori_Xtest, ori_ytrain, ori_ytest = train_test_split(ori_X, ori_y, test_size = 0.2, random_state = 2023)

new_df = pd.concat([non_fraud, fraud])
new_df = new_df.sample(frac = 1, random_state = 2023)
print(len(new_df))
print(new_df.describe())

new_df['Class'].plot.hist()
plt.title('new df class proportion')
plt.show()


new_X = new_df.drop('Class', axis = 1)
new_y = new_df['Class']

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size = 0.3, random_state = 2023)


params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
lr_reg = GridSearchCV(LogisticRegression(solver = 'liblinear'), params)
lr_reg.fit(X_train, y_train)
pred = cross_val_predict(lr_reg, X_train, y_train, cv = 5, method = 'decision_function')

fpr, trp, thred = roc_curve(y_train, pred)

plt.figure()
plt.plot(fpr, trp)
plt.show()


skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in skf.split(ori_X, ori_y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = ori_X.iloc[train_index], ori_X.iloc[test_index]
    original_ytrain, original_ytest = ori_y.iloc[train_index], ori_y.iloc[test_index]

undersample_y_score = lr_reg.decision_function(original_Xtest)
precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)
plt.figure()
plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
rand_lr = RandomizedSearchCV(LogisticRegression(), params, n_iter=4)
for train_index, test_index in skf.split(ori_Xtrain, ori_ytrain):
    imbalanced_pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_lr) # SMOTE happens during Cross Validation not before..
    model = imbalanced_pipeline.fit(ori_Xtrain.iloc[train_index], ori_ytrain.iloc[train_index])
    best_est = rand_lr.best_estimator_
    prediction = best_est.predict(ori_Xtrain.iloc[test_index])

smote_prediction = best_est.predict(original_Xtest)
y_score = best_est.decision_function(original_Xtest)

plt.figure()
precision, recall, _ = precision_recall_curve(ori_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

