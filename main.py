from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from utils import *
import argparse
import mlflow
import mlflow.sklearn


def main(FILE_NAME):
    # Split the dataset
    Xtrain, Xtest, ytrain, ytest = read_data(FILE_NAME)
    print('total number of training samples are {}, the fraud is {}, proportion is {:.4f}'.format(len(ytrain), sum(ytrain == 1), sum(ytrain == 1) / len(ytrain)))
    print('total number of testing samples are {}, the fraud is {}, proportion is {:.4f}, '.format(len(ytest), sum(ytest == 1), sum(ytest == 1) / len(ytest)))

    # Define the model, and if use SMOTE or not
    #model_names = [['logistic_regression', True], ['random_forest', True], ['gradient_boost', True],
    #               ['logistic_regression', False], ['random_forest', False], ['gradient_boost', False]]
    model_names = [['logistic_regression', True], ['random_forest', True], ['logistic_regression', False], ['random_forest', False]]
    nrow = len(model_names[0])
    ncol = len(model_names) // 2
    fig, axis = plt.subplots(nrow, ncol)
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2023)
    EXPERIMENT_NAME = 'Affect of SMOTE to LR, RF, and GBDT'
    # Run the models, and draw the precision-recall plot for each model
    for i, (model_name, use_smt) in enumerate(model_names):
        print('[INFO]: running {} ...'.format(model_name))
        preprocessor = get_preprocessor()
        model = get_model(MODEL_NAME=model_name,  CV_STRATEGY = stratified_kfold)

        # If use smote upsampling, then include it, else not
        if use_smt:
            smote = SMOTE(sampling_strategy='minority', random_state = 2023)
            pipeline = imbpipeline(steps=[('preprocesor', preprocessor),
                                          ('smote', smote),
                                          ('model', model)])
        else:
            pipeline = imbpipeline(steps = [('preprocesor', preprocessor),
                                            ('model', model)])


        pipeline.fit(Xtrain, ytrain)

        RUN_NAME = 'MODEL: {}, SMOTE: {}'.format(model_name, use_smt)
        report = classification_report(ytrain, pipeline.predict(Xtrain), output_dict= True)
        print(RUN_NAME)
        print(report)

        # Log the parameters using MLflow
        mlflow.set_experiment(experiment_name= EXPERIMENT_NAME)
        with mlflow.start_run(run_name= RUN_NAME):
            mlflow.sklearn.log_model(model.best_estimator_, "model")
            mlflow.log_metric('Precision', report['1']['precision'])
            mlflow.log_metric('Recall', report['1']['recall'])
            mlflow.log_metric('F1 score', report['1']['f1-score'])

        # Draw the precision-recall plots
        y_pred = pipeline.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, y_pred[:, 1])
        axis[i // ncol, i % ncol].set_title(EXPERIMENT_NAME)
        draw_plot(axis[i // ncol, i % ncol], precision, recall)
    plt.show()

    print('[INFO]: complete, exit algorithms...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'fraud-detector')
    parser.add_argument('--filename', type = str, default = 'creditcard.csv')
    args = parser.parse_args()
    filename = args.filename

    main(filename)