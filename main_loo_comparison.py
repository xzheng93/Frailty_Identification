import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import shap
from Utils import plot_cm, plot_roc, plot_roc_compare
from skopt import BayesSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
SAVE = False
n_iter = 15


def convert_label(value):
    if 0 <= value <= 1:
        return 0
    elif 2 <= value <= 14:
        return 1
    else:
        return None


# define the parameter space
def build_knn(train_X, train_y):
    opt = BayesSearchCV(KNeighborsClassifier(), {'n_neighbors': (1, 25), 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                                 'weights': ['uniform', 'distance']},
        n_iter=n_iter, cv=5, return_train_score=True, random_state=27)
    opt.fit(train_X, train_y)
    return opt


def build_nb(train_X, train_y):
    opt = BayesSearchCV(GaussianNB(), {
        'var_smoothing': (1e-11, 1e-7, 1e-10), }, n_iter=n_iter, cv=5, return_train_score=True, random_state=27)
    opt.fit(train_X, train_y)
    return opt


def build_xgboost(train_X, train_y):
    opt = BayesSearchCV(XGBClassifier(random_state=27), {'eta': (0.01, 1, 'log-uniform'), 'booster': ['gbtree', 'dart'],
                                                         'max_depth': (1, 15),  # integer valued parameter
                                                         'n_estimators': (100, 1000)},
        n_iter=n_iter, cv=5, return_train_score=True, random_state=27)
    opt.fit(train_X, train_y)
    return opt


def build_rf(train_X, train_y):
    opt = BayesSearchCV(RandomForestClassifier(random_state=27), {'min_samples_split': (5, 20),
                                                                  'max_leaf_nodes': (5, 20), 'max_depth': (1, 15),
                                                                  # integer valued parameter
                                                                  'n_estimators': (100, 1000, 10)},
        n_iter=n_iter, cv=5, return_train_score=True, random_state=27)
    opt.fit(train_X, train_y)
    return opt


def build_svm(train_X1, train_y1):
    opt = BayesSearchCV(SVC(random_state=27), {'C': (1, 250), 'gamma': (0.01, 10),
                                                               'probability': [True], 'kernel': ['rbf'],
                                               'degree': (1, 50), },
        n_iter=n_iter, cv=5, return_train_score=True, random_state=27)
    opt.fit(train_X, train_y)
    return opt


if __name__ == '__main__':

    # load gait data and remove unwanted gait features
    gait_data = pd.read_csv('./data/gait results.csv')
    redundant_features = ['trail', 'Distance', 'LyapunovRosen_ML', 'LyapunovRosen_AP', 'LyapunovRosen_V',
                          'RelativeStrideVariability_V', 'RelativeStrideVariability_AP', 'RelativeStrideVariability_ML',
                          'RelativeStrideVariability_All', 'StrideTimeSeconds', 'HarmonicRatioP_V', 'HarmonicRatioP_ML',
                          'HarmonicRatioP_AP', 'LyapunovPerStrideRosen_V', 'LyapunovPerStrideRosen_ML',
                          'LyapunovPerStrideRosen_AP']
    gait_data.drop(columns=redundant_features, inplace=True)
    # Replace 0 with NaN
    gait_data.replace(0, np.nan, inplace=True)
    # get the averaged results base on id
    gait_data_aver = gait_data.groupby('Subjects').mean().reset_index()

    # load demographic data
    demo_data = pd.read_csv('./data/MBF-GENDER.csv')
    selected_columns = ['Subjects', 'Age', 'Gender (0M; 1F)', 'Height (cm)', 'Weight (kg)', 'BMI (kg/m^2)',
                        '% in Sedentary', '% in Light', '% in MVPA', 'FRAIL-NH', 'Aids']
    demo_data_selected = demo_data[selected_columns]

    # form the dataset
    dataset = pd.merge(demo_data_selected, gait_data_aver, on='Subjects', how='left')
    nan_count_per_row = dataset.isna().sum(axis=1)
    no_gait_index = np.where(nan_count_per_row >= 37)[0]
    # remove data without gait features data (no IMU data)
    dataset.drop(no_gait_index, inplace=True)
    # set the NH Frailty label to 0 (HC), 1 (frailty)
    dataset['FRAIL-NH'] = dataset['FRAIL-NH'].apply(convert_label)

    labels = dataset['FRAIL-NH'].to_numpy()
    features = dataset.drop(columns=['FRAIL-NH', 'Subjects'])
    subjects = dataset['Subjects'].to_numpy()

    ########################
    model_list = ['svm',  'xgboost', 'knn', 'nb', 'rf']
    classifiers = {'nb': build_nb, 'svm': build_svm, 'knn': build_knn, 'rf': build_rf, 'xgboost': build_xgboost}
    fpr_list, tpr_list, model_name_list, auc_list = [], [], [], []
    for model in model_list:
        print(f'######################## now is training model: {model} ########################')
        # pre-process the data
        if model == 'knn' or model == 'nb' or model == 'svm':
            processed_features = features.fillna(features.mean())
            transformer = preprocessing.MinMaxScaler().fit(processed_features)
            processed_features = transformer.transform(processed_features)
            kpca = KernelPCA(gamma=0.01, kernel='rbf', n_components=15).fit(processed_features)  # 91%
            processed_features = kpca.transform(processed_features)
            columns = [f'component_{i + 1}' for i in range(processed_features.shape[1])]
            processed_features = pd.DataFrame(processed_features, columns=columns)  # find the number of components  # explained_variance = np.var(kpca.transform(processed_features), axis=0)  # explained_variance_ratio = explained_variance / np.sum(explained_variance)  # cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        elif model == 'rf':
            processed_features = features.fillna(features.mean())
        elif model == 'xgboost':
            processed_features = features

        # define final evaluation metrics for LOO-CV
        true_labels = []
        predicted_labels = []
        predicted_probs = []
        test_list = []
        shap_list = []
        loo = LeaveOneOut()
        for i, (train_index, test_index) in enumerate(loo.split(processed_features)):
            train_X = processed_features.iloc[train_index]
            train_y = labels[train_index]
            test_X = processed_features.iloc[test_index]
            test_y = labels[test_index]
            opt = classifiers[model](train_X, train_y)
            '''
            Evaluation
            '''
            print("val. score: %s" % opt.best_score_)
            print("test score: %s" % opt.score(test_X, test_y))
            print("best params: %s" % opt.best_params_)

            y_pred = opt.predict(test_X)
            pred_probability = opt.predict_proba(test_X)

            predicted_probs.append(pred_probability[:, 1])
            predicted_labels.append(y_pred)
            true_labels.append(test_y)

        # Convert lists to arrays
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        predicted_probs = np.array(predicted_probs)

        conf_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
        plot_cm(conf_matrix, ['Non-Frailty', 'Pre-/Frailty'], f'Confusion Matrix of {model}')
        if SAVE:
            plt.savefig(f'./results/confusion_matrix_{model}.pdf')
        plt.show()

        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_probs)
        auc = plot_roc(fpr, tpr, f'ROC Curve of {model}')
        if SAVE:
            plt.savefig(f'./results/ROC_{model}.pdf')
        plt.show()

        model_name_list.append(model)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        auc_list.append(auc)

        # Extract values from confusion matrix
        TP = conf_matrix[1, 1]  # True Positives
        TN = conf_matrix[0, 0]  # True Negatives
        FP = conf_matrix[0, 1]  # False Positives
        FN = conf_matrix[1, 0]  # False Negatives

        # Compute metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Avoid division by zero
        precesion = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1 = 2 * (sensitivity * precesion) / (sensitivity + precesion)

        print(f"performance of {model} is")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precesion:.4f}")
        print(f"F1: {F1:.4f}")

    # plot the comparison of all ML models
    plot_roc_compare(fpr_list, tpr_list, model_name_list, auc_list, 'ROC comparison of machine learning')
    plt.savefig('./results/ROC comparison of machine learning.pdf', format='pdf')
    plt.show()
    print()

