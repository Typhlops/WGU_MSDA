import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
# one-hot encoding that maintains a list of encoded columns in global_encoded_columns
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


# Setting dependent variable for correlation search, displaying the strongest correlations on 'TotalCharge'
# corr_search on 'Initial_days' and subsequent variables is done after inspecting a heatmap
dep_var = 'TotalCharge'
with pd.option_context('display.max_rows', None):
    corr_search(dep_var)
corr_search('Initial_days', 0.03)
corr_search('ReAdmis_Yes', 0.03)
corr_search('Initial_admin_Emergency Admission', 0.03)
corr_search('Additional_charges', 0.03)
corr_search('Complication_risk_Low', 0.03)
corr_search('Complication_risk_Medium', 0.03)

# Saving cleaned data set
df_med.to_csv('medical_cleaned_prediction.csv')


# Sets model's dependent variable and independent variables.
# top_features are determined by the output of feature_selection and hyper_search (found below).
target_var = dep_var
y_0 = df_med[target_var]
X_0 = df_med.select_dtypes(exclude='object').drop([target_var], axis=1)
top_features = ['ReAdmis_Yes', 'Initial_days', 'Initial_admin_Emergency Admission', 'Complication_risk_High',
                'Complication_risk_Medium', 'Initial_admin_Observation Admission', 'Initial_admin_Elective Admission',
                'BackPain_Yes', 'Arthritis_Yes', 'Anxiety_Yes', 'Additional_charges', 'Reflux_esophagitis_Yes',
                'Marital_Divorced', 'Children', 'TimeZone_America/Phoenix', 'HighBlood_Yes']
X_1 = df_med[top_features]

# Saving subset of clean data for independent and dependent variables
Z_0 = df_med[top_features + [target_var]]
Z_0.to_csv('medical_transformed_prediction.csv')


# Scales independent variables in X with a StandardScaler, MinMaxScaler, RobustScaler, or nothing. If split is True,
# the X and y inputs are split into training and test data that are saved to .csv files when save is True. Returns the
# scaled dataframe X_df (and the train/test split if applicable).
def scale_split(y, X, scl='standard', split=True, save=True, split_size=0.2, seed=16):
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
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=split_size, random_state=seed)
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
# features to those with a p-value <= alpha. Returns a dataframe of the sorted features and list of columns with
# p-value below alpha.
def feature_selection(y, X, alpha=0.05, output=True, df=df_med):
    best_feat = SelectKBest(f_regression, k='all')
    best_feat.fit(X, y)
    df_features = pd.DataFrame({'Score': best_feat.scores_, 'p-value': best_feat.pvalues_},
                               index=best_feat.feature_names_in_).sort_values(by='p-value')
    df_refine = df_features[df_features['p-value'] <= alpha]
    if output:
        with pd.option_context('display.max_rows', None):
            print(df_features)
        print(f"\nVIF for features with p-values <= {alpha}:")
        vif_print(df[df_refine.index])
    return df_features, df_refine.index


# Random forest regressor for target y ('TotalCharge' in this case) with independent variables X (see top_features)
def predictor_rf(y, X, n_est=100, m_feat=4, m_dep=8, split_frc=0.2, seed=16, state=25, save=True, output=True, plot=True):
    X_df, X_train, X_test, y_train, y_test = scale_split(y, X, 'none', True, save, split_frc, seed)

    rfr = RandomForestRegressor(n_estimators=n_est, max_features=m_feat, max_depth=m_dep, random_state=state)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    rsquare = r2_score(y_test, y_pred)
    rsquare_train = r2_score(y_train, rfr.predict(X_train))
    mse = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, rfr.predict(X_train))
    rmse = np.sqrt(mse)
    rmse_train = np.sqrt(mse_train)
    mae = mean_absolute_error(y_test, y_pred)
    mae_train = mean_absolute_error(y_train, rfr.predict(X_train))
    df_feat_import = pd.DataFrame({'feature_importance': rfr.feature_importances_}, index=X_df.columns)

    if output:
        print(f"\nR^2: {rsquare}")
        print(f"\nR^2 on train: {rsquare_train}")
        print(f"\nMSE: {mse}")
        print(f"\nMSE on train: {mse_train}")
        print(f"\nRMSE: {rmse}")
        print(f"\nRMSE on train: {rmse_train}")
        print(f"\nMAE: {mae}")
        print(f"\nMAE on train: {mae_train}")
        print(f"\nFeature importances:\n{df_feat_import}")
        print(f"\nCross validation scores: {cross_val_score(rfr, X_train, y_train, cv=5)}")

    if plot:
        y_pred_all = rfr.predict(X)
        residuals = y - y_pred_all
        sns.scatterplot(x=y_pred_all, y=residuals)
        plt.xlabel('Predicted value of MonthlyCharge')
        plt.ylabel('Residuals')
        plt.title('Scatter plot of residuals vs predicted values')
        plt.show()

        sm.qqplot(residuals)
        plt.title('QQ plot of residuals')
        plt.show()

    return rfr, rsquare, mse, mae


param_dict = {"n_estimators": [100, 200, 400], "max_features": [4, 5, 6, 7, 8], "max_depth": [8, None]}
# Adaptation of GridSearchCV hyperparametric tuning for the random forest regressor using the param_dict above as the
# default param_grid dictionary. Prints optimal parameters and model statistics such as MSE and R^2. Returns optimal 
# parameters, R^2, and MSE.
def hyper_param(y, X, parameters=param_dict, save=False, split_frc=0.2, seed=16, state=25, verbosity=4):
    X_df, X_train, X_test, y_train, y_test = scale_split(y, X, 'none', True, save, split_frc, seed)
    rf_start = RandomForestRegressor(random_state=state)

    rf_grid = GridSearchCV(estimator=rf_start, param_grid=parameters, scoring='neg_mean_squared_error',
                           cv=5, n_jobs=-1, verbose=verbosity)
    rf_grid.fit(X_train, y_train)

    opt_params = rf_grid.best_params_
    opt_rf = rf_grid.best_estimator_
    y_pred = opt_rf.predict(X_test)
    rsquare = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    # df_results = pd.DataFrame(rf_grid.cv_results_)
    print(f"\nBest parameters: {opt_params}")
    print(f"\nR^2: {rsquare}")
    print(f"\nMSE: {mse}")
    print(f"\nRMSE: {rmse}")
    print(f"\nMAE: {mae}")
    return opt_params, rsquare, mse


# Hyperparametric tuning looping over which features are used as predictor variables in the random forest regressor.
# X_columns defaults to top_features defined above (a list of column names in the dataframe). top_features are in
# ascending order of p-values as determined by an F-test (f_regression) with a ceiling of p = 0.05. Each loop
# increments the final index of X_columns, so custom input for X_columns should be sorted by the desired order for
# them to be added. Returns a sorted dictionary of the form 'num_features': (MSE, R^2, GridSearchCV().best_params_).
def hyper_feature(y, X_columns=top_features, sv=False, splt=0.2, sd=16, ste=25, vb=3, df=df_med):
    hyper_rf_dict = {}
    for i in range(1, len(X_columns) + 1):
        X_trial = df[X_columns[:i]]
        print(f"\nBeginning loop for columns: {X_columns[:i]}:\n")
        opt_params, rsquare, mse = hyper_param(y, X_trial, save=sv, split_frc=splt, seed=sd, state=ste, verbosity=vb)
        hyper_rf_dict[i] = (mse, rsquare, opt_params)
    sorted_hyper_rf_dict = dict(sorted(hyper_rf_dict.items(), key=lambda z: z[1][0]))
    return sorted_hyper_rf_dict


# Looking for best features using X_0 as "kitchen sink" initial set of variables
df_features, opt_indp_var = feature_selection(y_0, X_0, 0.05)
print(opt_indp_var)

# Looking for model statistics and optimal parameters using X_0 set of variables
#opt_params0, r20, mse0 = hyper_param(y_0, X_0)

# Searching for best subset of opt_indp_var independent variables to minimize MSE
# Long runtime
#print(hyper_feature(y_0, opt_indp_var))

# Having removed 'Item1' from top features (forming X_1 above) we look for the optimal random forest parameters
#opt_params, r21, mse1 = hyper_param(y_0, X_1)

# Pass the optimal parameters opt_params to a function that creates a random forest regression model
'''
c_e = opt_params['n_estimators']
c_f = opt_params['max_features']
c_d = opt_params['max_depth']
predictor_rf(y_0, X_1, n_est=c_e, m_feat=c_f, m_dep=c_d, split_frc=0.2, seed=16, state=25, save=True, output=True, plot=True)
'''

# For convenience, substituting those values directly without running hyper_param
predictor_rf(y_0, X_1, n_est=400, m_feat=8, m_dep=None, split_frc=0.2, seed=16, state=25, save=True, output=True, plot=True)



'''
print(hyper_feature(y_0, opt_indp_var)) output:

{16: (6881.381118954705, 0.9985461654130986, {'max_depth': None, 'max_features': 8, 'n_estimators': 400}), 
17: (7188.651690605722, 0.9984812481273269, {'max_depth': None, 'max_features': 8, 'n_estimators': 400}), 
13: (8048.991718179834, 0.9982994834398373, {'max_depth': None, 'max_features': 7, 'n_estimators': 400}), 
12: (8064.962682078371, 0.9982961092422308, {'max_depth': None, 'max_features': 7, 'n_estimators': 400}), 
14: (8201.059096421575, 0.9982673560499711, {'max_depth': None, 'max_features': 8, 'n_estimators': 400}), 
15: (8233.448233211548, 0.998260513172576, {'max_depth': None, 'max_features': 8, 'n_estimators': 400}), 
11: (8683.26079279174, 0.9981654809333445, {'max_depth': None, 'max_features': 6, 'n_estimators': 400}), 
10: (10927.979890339488, 0.9976912374340411, {'max_depth': None, 'max_features': 6, 'n_estimators': 400}), 
9: (11210.996064050794, 0.9976314443932429, {'max_depth': 8, 'max_features': 7, 'n_estimators': 400}), 
8: (11727.434163585538, 0.9975223361258589, {'max_depth': 8, 'max_features': 6, 'n_estimators': 400}), 
7: (12921.868175603193, 0.9972699871499159, {'max_depth': 8, 'max_features': 6, 'n_estimators': 400}), 
5: (12936.820136350689, 0.9972668282378747, {'max_depth': 8, 'max_features': 4, 'n_estimators': 400}), 
6: (12941.472671077345, 0.9972658452933486, {'max_depth': 8, 'max_features': 5, 'n_estimators': 400}), 
4: (12978.497433192339, 0.9972580230438897, {'max_depth': 8, 'max_features': 4, 'n_estimators': 400}), 
3: (51681.32321405701, 0.9890812478067134, {'max_depth': 8, 'max_features': 4, 'n_estimators': 200}), 
2: (119504.51875014645, 0.9747521900550736, {'max_depth': 8, 'max_features': 4, 'n_estimators': 200}), 
1: (1322424.9265026082, 0.7206102868747785, {'max_depth': 8, 'max_features': 4, 'n_estimators': 400})}
'''



