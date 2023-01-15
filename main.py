from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from utils import *

# Define the dataset
FILE_NAME = 'creditcard.csv'
# Split the dataset
Xtrain, Xtest, ytrain, ytest = read_data(FILE_NAME)

# Define the steps in the pipeline
preprocessor = get_preprocessor()
smote = SMOTE(sampling_strategy='minority')

# Define the model, and if use SMOTE or not
model_names = [('logistic_regression', True), ('random_forest', True), ('logistic_regression', False), ('random_forest', False)]


fig, axis = plt.subplots(2, len(model_names) // 2)

# Run the models, and draw the precision-recall plot for each model
for i, (model_name, use_smote) in enumerate(model_names):
    print('[INFO]: running {} ...'.format(model_name))
    model = get_model(MODEL_NAME = model_name, n_iter = 1)

    # If use smote upsampling, then include it, else not
    if use_smote:
        pipeline = make_pipeline(preprocessor, smote, model)
        pipeline.fit(Xtrain, ytrain)
    else:
        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(Xtrain, ytrain)

    best_est = model.best_estimator_
    if model_name == 'random_forest':
        y_score = best_est.predict_proba(preprocessor.transform(Xtest))
        y_score = y_score[:, 1]
    elif model_name == 'logistic_regression':
        y_score = best_est.decision_function(preprocessor.transform(Xtest))
    elif model_name == 'gradient_boost':
        y_score = best_est.decision_function(preprocessor.transform(Xtest))
    precision, recall, _ = precision_recall_curve(ytest, y_score)

    axis[i].set_title(model_name + str(use_smote))
    draw_plot(axis[i], precision, recall)
plt.show()