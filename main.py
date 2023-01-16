from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from utils import *
import argparse

def main(FILE_NAME):
    # Split the dataset
    Xtrain, Xtest, ytrain, ytest = read_data(FILE_NAME)
    print('total number of training samples are {}, the fraud is {}, proportion is {:.4f}'.format(len(ytrain), sum(ytrain == 1), sum(ytrain == 1) / len(ytrain)))
    print('total number of testing samples are {}, the fraud is {}, proportion is {:.4f}, '.format(len(ytest), sum(ytest == 1), sum(ytest == 1) / len(ytest)))

    # Define the model, and if use SMOTE or not
    model_names = [['logistic_regression', True], ['random_forest', True], ['gradient_boost', True],
                   ['logistic_regression', False], ['random_forest', False], ['gradient_boost', False]]

    nrow = len(model_names[0])
    ncol = len(model_names) // 2
    fig, axis = plt.subplots(nrow, ncol)
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2023)
    # Run the models, and draw the precision-recall plot for each model
    for i, (model_name, use_smt) in enumerate(model_names):
        print('[INFO]: running {} ...'.format(model_name))
        preprocessor = get_preprocessor()
        model, params = get_model(MODEL_NAME=model_name)

        # If use smote upsampling, then include it, else not
        if use_smt:
            smote = SMOTE(sampling_strategy='minority', random_state = 2023)
            pipeline = imbpipeline(steps=[('preprocesor', preprocessor),
                                          ('smote', smote),
                                          ('model', model)])
        else:
            pipeline = imbpipeline(steps = [('preprocesor', preprocessor),
                                            ('model', model)])

        grid_search = GridSearchCV(estimator = pipeline, param_grid = params, cv=stratified_kfold)
        grid_search.fit(Xtrain, ytrain)

        print('MODEL: {}, SMOTE: {}'.format(model_name, use_smt))
        print(classification_report(ytrain, grid_search.predict(Xtrain)))

        # Draw the precision-recall plots
        y_pred = grid_search.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, y_pred[:, 1])
        figure_title = 'MODEL: {}, SMOTE: {}'.format(model_name, use_smt)
        axis[i // ncol, i % ncol].set_title(figure_title)
        draw_plot(axis[i // ncol, i % ncol], precision, recall)
    plt.show()

    print('[INFO]: complete, exit algorithms...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'fraud-detector')
    parser.add_argument('--filename', type = str, default = 'creditcard.csv')
    args = parser.parse_args()
    filename = args.filename

    main(filename)