import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, \
    roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

df_med = pd.read_csv('data/Medical_Data_Set/medical_clean.csv')
original_df_med = pd.read_csv('data/Medical_Data_Set/medical_clean.csv')
pd.set_option('display.max_columns', None)

# When True, displays extensive summary statistics and plots. Output streamlined when False.
run_verbose = False

all_columns = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
               'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job',
               'Children', 'Age', 'Income', 'Marital', 'Gender', 'ReAdmis',
               'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
               'Soft_drink', 'Initial_admin', 'HighBlood', 'Stroke',
               'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes',
               'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
               'Reflux_esophagitis', 'Asthma', 'Services', 'Initial_days',
               'TotalCharge', 'Additional_charges', 'Item1', 'Item2', 'Item3', 'Item4',
               'Item5', 'Item6', 'Item7', 'Item8']

identifier_cat_columns = ['CaseOrder', 'Customer_id', 'Interaction', 'UID']

large_cat_columns = ['City', 'County', 'Zip', 'Job']

small_cat_columns = ['State', 'Area', 'TimeZone', 'Marital', 'Gender', 'Initial_admin', 'Complication_risk', 'Services']

yes_no_columns = ['ReAdmis', 'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
                  'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma']

item1_to_8_columns = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8']

small_int_columns = ['Children', 'Age', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp']

continuous_columns = ['Lat', 'Lng', 'Population', 'Income', 'VitD_levels', 'Initial_days', 'TotalCharge',
                      'Additional_charges']


# Creates a new column 'Zip_int64' to back up the old 'Zip' values while adjusting current 'Zip' values to strings with
# five digits.
def zip_to_str(zip_col='Zip', df=df_med):
    df['Zip_int64'] = df[zip_col]
    df[zip_col] = df[zip_col].astype('str')
    for i in range(5):
        df[zip_col].mask(df[zip_col].str.len() == i, '0' * (5 - i) + df[zip_col], inplace=True)
    print(
        f"Verifying number of entries in 'Zip' with number of digits other than 5: "
        f"{len(df.loc[df['Zip'].str.len() != 5, 'Zip'])}\n")
    print(
        f"Verifying number of entries in 'Zip' with number of digits exactly 5: "
        f"{len(df.loc[df['Zip'].str.len() == 5, 'Zip'])}\n")


# Changing 'CaseOrder' and 'Zip' to strings, also verifying there are no duplicates or nulls
df_med['CaseOrder'] = df_med['CaseOrder'].astype('str')
zip_to_str()
print(f"Checking for columns with null values: {list(df_med.columns[df_med.isna().sum() > 0])}\n")
print("Verifying there are no duplicate entries ('False' indicates not a duplicate):")
print(df_med.duplicated(keep=False).value_counts())
print("\n")


# Dataframe description and value counts
def inspect_data(columns, df=df_med):
    for col in columns:
        if (df[col].dtype == 'int64') or (df[col].dtype == 'float64'):
            print(f"\nNumber of unique values: {len(df[col].unique())}")
            print(df[col].describe())
        else:
            print(df[col].describe())
            print(df[col].value_counts())
            print("\n")


# Searches for outliers by IQR and z-scores (defaults to |z| > 3.0) with optional z-score histogram plot
def outlier_search(columns, plots=True, z_bound=3.0, df=df_med):
    df_outliers_dict = {}
    df_zscore_outl_dict = {}
    for column in columns:
        col_stats = df[column].describe()
        q25 = col_stats['25%']
        q75 = col_stats['75%']
        lower_bound = q25 - 1.5 * (q75 - q25)
        upper_bound = q75 + 1.5 * (q75 - q25)
        df_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        col_zscore = column + '_zscore'
        with pd.option_context("mode.chained_assignment", None):
            df_outliers[col_zscore] = stats.zscore(df[column])
        df_zscore = pd.DataFrame(stats.zscore(df[column]))
        df_zscore_outl = df_zscore[abs(df_zscore[column]) > z_bound]
        print("----------------------")
        print(f"{column}:")
        print(col_stats)
        print("\nZ-scores:")
        print(df_zscore.describe())
        print(f"\nIQR test for outliers has a lower bound of {round(lower_bound, 3)} and an upper bound"
              f" of {round(upper_bound, 3)}")
        print(f"Z-scores have a lower bound of {-1 * z_bound} and an upper bound of {z_bound}\n")
        if df_outliers.empty and df_zscore.empty:
            print(f"There are no outliers in the column {column}.")
        else:
            print(f"By IQR, there are {len(df_outliers)} outliers.")
            print(df_outliers[[column, col_zscore]])
            print(f"\nBy z-score, there are {len(df_zscore_outl)} outliers.")
            print(df_zscore_outl)
        df_outliers_dict[column] = df_outliers
        df_zscore_outl_dict[column] = df_zscore_outl
        if plots:
            plt.hist(df_zscore)
            plt.xlabel(column + ' z-score')
            plt.ylabel('Frequency')
            plt.show()
        print("----------------------\n")
    return df_outliers_dict, df_zscore_outl_dict


if run_verbose:
    inspect_data(df_med.columns)
    outlier_search(small_int_columns)
    outlier_search(continuous_columns)
    outlier_search(item1_to_8_columns)


global_encoded_columns = []
# one hot encoding that maintains a list of encoded columns in global_encoded_columns
def one_hot_encoder(columns, df=df_med):
    for column in columns:
        if column in yes_no_columns:
            df_one_hot_col = pd.get_dummies(df[column], drop_first=True).astype('int32')
        else:
            df_one_hot_col = pd.get_dummies(df[column]).astype('int32')
        for col in df_one_hot_col.columns:
            col_name = f'{column}_' + col
            if col_name not in global_encoded_columns:
                global_encoded_columns.append(col_name)
                df[col_name] = df_one_hot_col[col]


# Any of 'State', 'Job', 'City', 'County', 'Zip' created a large number of additional columns that were found to have
# little relevance. The remaining categorical variables (excluding irrelevant identifiers) are to be one-hot encoded.
columns_to_encode = ['Area', 'Gender', 'Marital', 'TimeZone', 'Initial_admin', 'Complication_risk', 'Services',
                     'ReAdmis', 'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
                     'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma']
one_hot_encoder(columns_to_encode)
print("Verifying appended columns in dataframe:")
print(df_med.columns)
print("\nVerifying new columns have expected values:")
print(df_med['Diabetes_Yes'].value_counts())
print(df_med['Initial_admin_Emergency Admission'].value_counts())
print(f"\nColumns encoded: {global_encoded_columns}\n")


# Prints correlation matrix values restricted to a specific variable ('column') sorted in descending order of their
# absolute value. Optional threshold parameter 'thresh' defaults to -1.0 and restricts correlated variables to
# those above it.
def corr_search(column, thresh=-1.0, df=df_med):
    sorted_corr = df.select_dtypes(exclude=['object']).corr()[column].sort_values(ascending=False, key=lambda x: abs(x))
    print(sorted_corr[abs(sorted_corr) >= thresh])
    print("\n")


# Setting dependent variable for correlation search, displaying the strongest correlations on 'ReAdmis_Yes'
# corr_search on 'Initial_days' and subsequent variables is done after inspecting a heatmap
dep_var_str = 'ReAdmis'
dep_var = dep_var_str + '_Yes'
with pd.option_context('display.max_rows', None):
    corr_search(dep_var)
corr_search('Initial_days', 0.03)

# Saving cleaned data set
df_med.to_csv('medical_cleaned_classification.csv')


# Sets model's dependent variable and independent variables.
# top_features are determined by the output of feature_selection and hyper_search (found below).
target_var = dep_var
y_0 = df_med[target_var]
X_0 = df_med.select_dtypes(exclude='object').drop([target_var], axis=1)
top_features = ['Initial_days', 'Services_CT Scan', 'Children', 'Marital_Divorced', 'Services_Intravenous']
X_1 = df_med[top_features]

# Saving subset of clean data for independent and dependent variables
Z_0 = df_med[top_features + [target_var]]
Z_0.to_csv('medical_transformed_classification.csv')


# Scales independent variables in X with a StandardScaler, MinMaxScaler, or RobustScaler. If split is True,
# the X and y inputs are split into training and test data that are saved to .csv files. Returns the scaled dataframe
# X_df (and the train/test split if applicable).
def scale_split(y, X, scl='standard', split=True, save=True, split_size=0.3, seed=16):
    if scl == 'standard':
        scaler = StandardScaler()
        X_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scl == 'minmax':
        scaler = MinMaxScaler()
        X_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scl == 'robust':
        scaler = RobustScaler()
        X_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scl == 'none':
        X_df = X
    else:
        print(f"Invalid choice of scaler {scl}. Exiting.")
        return

    if split:
        X_train, X_test, y_train, y_test = (
            train_test_split(X_df, y, test_size=split_size, random_state=seed, stratify=y))
        if save:
            X_train.to_csv('X_train.csv')
            X_test.to_csv('X_test.csv')
            y_train.to_csv('y_train.csv')
            y_test.to_csv('y_test.csv')
        return X_df, X_train, X_test, y_train, y_test
    else:
        return X_df


# Prints the variance inflation factors for the independent variable columns in X
def vif_print(X):
    vif_data = pd.DataFrame({'Feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]}).sort_values(by='VIF', ascending=False)
    with pd.option_context('display.max_rows', None):
        print("\n", vif_data)


# Uses SelectKBest to find the best features sorted by p-values (from an F-test). Uses alpha to restrict the sorted
# features to those with a p-value <= alpha. Returns a dataframe of the sorted features.
def feature_selection(y, X, alpha=0.05, output=True, df=df_med):
    best_feat = SelectKBest(f_classif, k='all')
    best_feat.fit(X, y)
    df_features = pd.DataFrame({'Score': best_feat.scores_, 'p-value': best_feat.pvalues_},
                               index=best_feat.feature_names_in_).sort_values(by='p-value')
    df_refine = df_features[df_features['p-value'] <= alpha]
    if output:
        with pd.option_context('display.max_rows', None):
            print(df_features)
        print(f"\nVIF for features with p-values <= {alpha}:")
        vif_print(df[df_refine.index])
    return df_features


# KNN classifier for target y ('ReAdmis_Yes' in this case) with independent variables X (see top_features)
def classifier_knn(y, X, scl='standard', nbors=40, split_frc=0.3, seed=16, vif=False, save=True, output=True, roc_plot=True):
    X_df, X_train, X_test, y_train, y_test = scale_split(y, X, scl, True, save, split_frc, seed)

    knn = KNeighborsClassifier(n_neighbors=nbors)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred_knn)
    accuracy_train = accuracy_score(y_train, knn.predict(X_train))
    auc = roc_auc_score(y_test, y_pred_prob_knn)

    if output:
        print(f"\nAccuracy: {accuracy}")
        print(f"\nAccuracy on train: {accuracy_train}")
        print(f"\nConfusion matrix: \n{confusion_matrix(y_test, y_pred_knn)}")
        print(f"\nCross validation scores: {cross_val_score(knn, X_train, y_train, cv=8)}")
        print(f"\nClassification report: \n{classification_report(y_test, y_pred_knn)}")
        print(f"\nArea under the curve: {auc}\n")

    if roc_plot:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_knn)
        plt.scatter(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for KNN model')
        plt.show()

    if vif:
        vif_print(X_df)

    return accuracy, auc, knn


# Creates multiple KNN models with varying values of n_neighbors from 1 to n to determine the one with the highest
# accuracy and the one with the highest AUC value. Returns dictionaries of its results, the optimal accuracy and AUC
# values, and the associated value of n_neighbors for those accuracy/AUC values.
def knn_num_search(n, y, X, scaling='standard', plot=True, output=True, split_frc=0.3, seed=16, save=False):
    knn_dict_acc = {}
    acc_num_opt = -1
    acc_opt = 0
    knn_dict_auc = {}
    auc_num_opt = -1
    auc_opt = 0
    for num in range(1, n+1):
        acc, auc, model = classifier_knn(y, X, scaling, num, split_frc, seed, False, save, False, False)
        knn_dict_acc[num] = acc
        knn_dict_auc[num] = auc
        if acc > acc_opt:
            acc_num_opt = num
            acc_opt = acc
        if auc > auc_opt:
            auc_num_opt = num
            auc_opt = auc
    sorted_dict_acc = dict(sorted(knn_dict_acc.items(), key=lambda item: item[1], reverse=True))
    sorted_dict_auc = dict(sorted(knn_dict_auc.items(), key=lambda item: item[1], reverse=True))
    if output:
        print(f"KNN model with 1 to {n} neighbors accuracies in descending order: \n{sorted_dict_acc}\n")
        print(f"Maximum occurred at n_neighbors = {acc_num_opt} with accuracy {knn_dict_acc[acc_num_opt]}\n")
        print(f"KNN model with 1 to {n} neighbors areas under curve in descending order: \n{sorted_dict_auc}\n")
        print(f"Maximum occurred at n_neighbors = {auc_num_opt} with area {knn_dict_auc[auc_num_opt]}\n")
    if plot:
        plt.scatter(x=list(knn_dict_acc.keys()), y=np.array(list(knn_dict_acc.values())))
        plt.xlabel('n_neighbors')
        plt.ylabel('Accuracy')
        plt.title('KNN accuracy vs n_neighbors')
        plt.show()
        plt.scatter(x=list(knn_dict_auc.keys()), y=np.array(list(knn_dict_auc.values())))
        plt.xlabel('n_neighbors')
        plt.ylabel('Area under curve')
        plt.title('KNN area under curve vs n_neighbors')
        plt.show()
    return knn_dict_acc, acc_num_opt, acc_opt, knn_dict_auc, auc_num_opt, auc_opt


# Hyperparametric search for the KNN models with the highest accuracy and AUC on the basis of features and number of
# nearest neighbors. Features are added sequentially by lowest p-value (retrieved from feature_selection's output)
# until the limit set by max_features has been reached. max_n is the highest value of n_neighbors used
# by knn_num_search. Produces scatter plots if desired and returns dictionaries of its search results.
def hyper_search(max_features=10, max_n=40, alpha=0.05, sclg='standard', split_f=0.3, seed=16, save=False, plots=True, target='ReAdmis_Yes', verbose=True, df=df_med):
    X_00 = df.select_dtypes(exclude='object').drop([target], axis=1)
    y = df[target]
    df_feat = feature_selection(y, X_00, alpha, verbose)
    var_tc = df_feat.index[0]
    acc_dict, acc_dict_tc, auc_dict, auc_dict_tc = {}, {}, {}, {}
    max_acc_columns, max_acc_columns_tc, max_auc_columns, max_auc_columns_tc = [], [], [], []
    max_acc, max_acc_tc, max_auc, max_auc_tc = 0, 0, 0, 0
    max_acc_adj, max_acc_adj_tc, max_auc_adj, max_auc_adj_tc = 0, 0, 0, 0
    for i in range(2, max_features+2):
        columns = list(df_feat.index[1:i])
        columns_tc = list([var_tc] + columns)
        X = df[columns]
        X_tc = df[columns_tc]
        knn_dict_acc, acc_num_opt, acc_opt, knn_dict_auc, auc_num_opt, auc_opt = (
            knn_num_search(max_n, y, X, sclg, False, verbose, split_f, seed, save))
        if acc_opt > max_acc:
            max_acc = acc_opt
            max_acc_adj = acc_num_opt
            max_acc_columns = columns
        if auc_opt > max_auc:
            max_auc = auc_opt
            max_auc_adj = auc_num_opt
            max_auc_columns = columns
        knn_dict_acc_tc, acc_num_opt_tc, acc_opt_tc, knn_dict_auc_tc, auc_num_opt_tc, auc_opt_tc = (
            knn_num_search(max_n, y, X_tc, sclg, False, verbose, split_f, seed, save))
        if acc_opt_tc > max_acc_tc:
            max_acc_tc = acc_opt_tc
            max_acc_adj_tc = acc_num_opt_tc
            max_acc_columns_tc = columns_tc
        if auc_opt_tc > max_auc_tc:
            max_auc_tc = auc_opt_tc
            max_auc_adj_tc = auc_num_opt_tc
            max_auc_columns_tc = columns_tc
        acc_dict[tuple(columns)] = (acc_num_opt, acc_opt)
        auc_dict[tuple(columns)] = (auc_num_opt, auc_opt)
        acc_dict_tc[tuple(columns_tc)] = (acc_num_opt_tc, acc_opt_tc)
        auc_dict_tc[tuple(columns_tc)] = (auc_num_opt_tc, auc_opt_tc)
    print(f"\nOptimal accuracy of {max_acc} occurred with n_neighbors = {max_acc_adj} and columns:\n{max_acc_columns}\n"
          f"With 'TotalCharge' included, optimal accuracy is {max_acc_tc} and "
          f"n_neighbors = {max_acc_adj_tc} and columns:\n{max_acc_columns_tc}\n"
          f"Optimal AUC of {max_auc} occurred with n_neighbors = {max_auc_adj} and columns:\n{max_auc_columns}\n"
          f"With 'TotalCharge' included, optimal AUC is {max_auc_tc} and "
          f"n_neighbors = {max_auc_adj_tc} and columns:\n{max_auc_columns_tc}\n")
    if plots:
        plt.scatter(x=[len(k) for k in acc_dict.keys()], y=np.array([s for r, s in acc_dict.values()]))
        plt.xlabel('Number of features (adding next lowest p-value from F-test)')
        plt.ylabel('Accuracy')
        plt.title('KNN accuracy vs number of features')
        plt.show()
        plt.scatter(x=[len(k) for k in acc_dict_tc.keys()], y=np.array([s for r, s in acc_dict_tc.values()]))
        plt.xlabel('Number of features (adding next lowest p-value from F-test)')
        plt.ylabel('Accuracy')
        plt.title(f'KNN accuracy vs number of features with {var_tc} included')
        plt.show()
        plt.scatter(x=[len(k) for k in auc_dict.keys()], y=np.array([s for r, s in auc_dict.values()]))
        plt.xlabel('Number of features (adding next lowest p-value from F-test)')
        plt.ylabel('Area under curve')
        plt.title('KNN AUC vs number of features')
        plt.show()
        plt.scatter(x=[len(k) for k in auc_dict_tc.keys()], y=np.array([s for r, s in auc_dict_tc.values()]))
        plt.xlabel('Number of features (adding next lowest p-value from F-test)')
        plt.ylabel('Area under curve')
        plt.title(f'KNN AUC vs number of features with {var_tc} included')
        plt.show()
    return acc_dict, acc_dict_tc, auc_dict, auc_dict_tc


#hyper_search(10, 40, 0.05, 'minmax')
#knn_num_search(40, y_0, X_1, 'minmax', True, True)
classifier_knn(y_0, X_1, 'minmax', 8)

