import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import shap
from Utils import plot_cm, plot_roc
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
SAVE = True

def convert_label(value):
    if 0 <= value <= 1:
        return 0
    elif 2 <= value <= 14:
        return 1
    else:
        return None


if __name__ == '__main__':

    # load gait data and remove unwanted gait features
    gait_data = pd.read_csv('./data/gait results.csv')
    redundant_features = ['trail', 'Distance', 'LyapunovRosen_ML', 'LyapunovRosen_AP',
                          'LyapunovRosen_V', 'RelativeStrideVariability_V', 'RelativeStrideVariability_AP',
                          'RelativeStrideVariability_ML', 'RelativeStrideVariability_All',
                          'StrideTimeSeconds', 'HarmonicRatioP_V', 'HarmonicRatioP_ML',
                          'HarmonicRatioP_AP',
                          'LyapunovPerStrideRosen_V', 'LyapunovPerStrideRosen_ML',
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
    # define final evaluation metric
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    shap_list = []
    test_list = []
    loo = LeaveOneOut()
    for i, (train_index, test_index) in enumerate(loo.split(features)):
        train_X = features.iloc[train_index]
        train_y = labels[train_index]
        test_X = features.iloc[test_index]
        test_y = labels[test_index]

        opt = BayesSearchCV(XGBClassifier(random_state=27), #scale_pos_weight
            {
                'eta': (0.01, 1, 'log-uniform'),
                'booster': ['gbtree', 'dart'],
                'max_depth': (1, 15),  # integer valued parameter
                'n_estimators': (100, 1000)
            },
            n_iter=15,
            cv=5,
            return_train_score=True,
            random_state=27
        )

        opt.fit(train_X, train_y)

        '''
        Evaluation
        '''
        print("val. score: %s" % opt.best_score_)
        print("test score: %s" % opt.score(test_X, test_y))
        print("best params: %s" % opt.best_params_)

        y_pred = opt.predict(test_X)
        pred_probability = opt.predict_proba(test_X)

        if y_pred != test_y:
            print(f'{subjects[test_index]} is worng')

        predicted_probs.append(pred_probability[:, 1])
        predicted_labels.append(y_pred)
        true_labels.append(test_y)
        '''
        SHAP
        '''
        explainer = shap.TreeExplainer(opt.best_estimator_)
        explanation = explainer(test_X)
        shap_values = explanation.values
        shap_list.append(shap_values)
        test_list.append(test_index)

    # Convert lists to arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    np.save('./results/true_labels.npy', true_labels)
    np.save('./results/predicted_labels.npy', predicted_labels)
    np.save('./results/predicted_probs.npy', predicted_probs)

    conf_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    plot_cm(conf_matrix, ['Non-Frailty', 'Pre-/Frailty'], f'Confusion Matrix')
    if SAVE:
        plt.savefig('./results/confusion_matrix.pdf')
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_probs)
    auc = plot_roc(fpr, tpr, f'ROC Curve')
    if SAVE:
        plt.savefig('./results/ROC.pdf')
    plt.show()

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
    F1 = 2 * (sensitivity * precesion) / (sensitivity+precesion)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precesion:.4f}")
    print(f"F1: {F1:.4f}")
    # combining results from all iterations
    test_set = test_list[0]
    shap_values = np.array(shap_list[0])
    for i in range(1, len(test_list)):
        test_set = np.concatenate((test_set, test_list[i]), axis=0)
        shap_values = np.concatenate((shap_values, np.array(shap_list[i])), axis=0)
    # bringing back variable names
    X_test = pd.DataFrame(features.iloc[test_set], columns=features.columns)
    # creating explanation plot for the whole experiment, the first dimension from shap_values indicate the class we are predicting (0=0, 1=1)
    shap.summary_plot(shap_values, X_test, max_display=10, show=False)
    if SAVE:
        plt.savefig('./results/Shap.pdf')
    plt.show()

    if SAVE:
        np.save('./results/shap_values.npy', shap_values)
        X_test.to_csv('./results/X_test.csv', index=False)

    abs_shap_values = np.sum(np.abs(shap_values),axis=0)
    sorted_indices = np.argsort(abs_shap_values)[::-1]
    keys = X_test.keys()
    top_10_A = abs_shap_values[sorted_indices[:10]]
    top_10_B = keys[sorted_indices[:10]]

    plt.bar(top_10_B, top_10_A, color='steelblue')
    # Adding labels and title cmap="Blues"
    plt.xlabel('Features')
    plt.ylabel('Accumulated ABS Shap values')
    plt.title('Feature importance')
    plt.xticks(rotation=45, ha='right')  # Rotate the labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    if SAVE:
        plt.savefig('./results/Feature importance.pdf')
    plt.show()
    print()

