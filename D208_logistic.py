import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

df_med = pd.read_csv('Medical_Data_Set/medical_clean.csv')
original_df_med = pd.read_csv('Medical_Data_Set/medical_clean.csv')
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
        f"Verifying number of entries in 'Zip' with number of digits other than 5: {len(df.loc[df['Zip'].str.len() != 5, 'Zip'])}\n")
    print(
        f"Verifying number of entries in 'Zip' with number of digits exactly 5: {len(df.loc[df['Zip'].str.len() == 5, 'Zip'])}\n")


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
        df_one_hot_col = pd.get_dummies(df[column], drop_first=True).astype('int32')
        for col in df_one_hot_col.columns:
            col_name = f'{column}_' + col
            if col_name not in global_encoded_columns:
                global_encoded_columns.append(col_name)
                df[col_name] = df_one_hot_col[col]


# 'Marital', 'TimeZone', and 'State' were found to have very low coefficients, but are not depicted due to the
# large number of additional columns they created. The remaining categorical variables are to be one hot encoded.
columns_to_encode = ['Area', 'Gender', 'Initial_admin', 'Complication_risk', 'Services',
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
corr_search(dep_var)
corr_search('Initial_days', 0.03)
corr_search('Additional_charges', 0.03)
print(df_med.select_dtypes(exclude=['object']).corr()[dep_var].sort_values(ascending=False, key=lambda x: abs(x)).index[1:19])


def univariate_plots(columns, df=df_med):
    for column in columns:
        plt.hist(df[column])
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f"Histogram of {column}")
        plt.show()


def bivariate_plots(columns, target_column=dep_var_str, df=df_med):
    for column in columns:
        plt.xlabel(column)
        plt.ylabel('Frequency')
        if ((df[column].dtype == 'float64') or
                ((df[column].dtype == 'int64') and (df[column].max() - df[column].min() > 10))):
            sns.kdeplot(data=df, x=column, hue=target_column)
            plt.title(f"Bivariate density plot of {column} grouped by {target_column}")
        else:
            sns.countplot(data=df, x=column, hue=target_column)
            plt.title(f"Bivariate histogram of {column} grouped by {target_column}")
        plt.show()


plot_columns = ['Initial_days', 'Children', 'Age', 'Population', 'Lat', 'Additional_charges', 'Complication_risk',
                'Initial_admin', 'BackPain', 'Arthritis', 'Anxiety', 'Asthma', 'Hyperlipidemia', 'Stroke',
                'Overweight', 'Services']
if run_verbose:
    univariate_plots([dep_var_str] + plot_columns)
    bivariate_plots(plot_columns)


# Creating a vector of 1s for a constant term.
df_med['model_constant'] = 1
df_save = df_med[['CaseOrder', 'ReAdmis_Yes', 'Initial_days', 'Children', 'Age', 'Population', 'Lat',
                  'Additional_charges', 'Complication_risk_Medium', 'Complication_risk_Low',
                  'Initial_admin_Emergency Admission', 'Initial_admin_Observation Admission', 'BackPain_Yes',
                  'Arthritis_Yes', 'Anxiety_Yes', 'Asthma_Yes', 'Hyperlipidemia_Yes', 'Stroke_Yes', 'Overweight_Yes',
                  'Services_MRI', 'Services_CT Scan', 'Services_Intravenous']]
df_save.to_csv(r'Medical_Data_Set/medical_transformed_logistic.csv')

# Initial model independent variables
model_indp_var = ['model_constant', 'Initial_days', 'Children', 'Age', 'Population', 'Lat', 'Additional_charges',
                  'Complication_risk_Medium', 'Complication_risk_Low', 'Initial_admin_Emergency Admission',
                  'Initial_admin_Observation Admission', 'BackPain_Yes', 'Arthritis_Yes', 'Anxiety_Yes', 'Asthma_Yes',
                  'Hyperlipidemia_Yes', 'Stroke_Yes', 'Overweight_Yes', 'Services_MRI', 'Services_CT Scan']
model_dep_var = 'ReAdmis_Yes'
X_0 = df_med[model_indp_var]
y_0 = df_med[model_dep_var]


# Logistic regression model for target y ('ReAdmis_Yes' in this case) with independent variables X (see model_indp_var)
def log_model(y, X):
    model = sm.Logit(y, X)
    results = model.fit()
    print(results.summary())
    print(f"\nAIC: {2 * (2 + results.df_model) - 2 * results.llf}")
    print(f"Predicted percentage of readmissions: "
          f"{np.round(results.predict()[results.predict() >= 0.5].sum() / len(results.predict()), 4)}")
    print(f"Accuracy: {accuracy_score(y, np.round(results.predict()))}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    model_sk = LogisticRegression(C=1.0, class_weight='balanced', max_iter=200, penalty='l1', solver='liblinear', random_state=5)
    model_sk.fit(X_train, y_train)
    print(f"Coefficients: {dict(zip(X.columns, model_sk.coef_[0]))}\n")
    y_pred = model_sk.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    validation = cross_val_score(model_sk, X_train, y_train, cv=8)
    metric_scores = classification_report(y_test, y_pred)
    print(f"Confusion matrix: \n{conf_matrix}\n")
    print(f"Cross validation scores: {validation}\n")
    print(f"Classification report: \n{metric_scores}\n")

    sns.heatmap(X.corr(), annot=True)
    plt.show()

    vif_data = pd.DataFrame({'feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]})
    print(vif_data)


# Adapted from DataCamp's "Dimensionality Reduction in Python" ch. 3 (Jeroen Boeye), accessed 2024.
# https://campus.datacamp.com/courses/dimensionality-reduction-in-python/feature-selection-ii-selecting-for-model-accuracy?ex=1
# RFE for reduction of initial model's independent variables
def rfe_review(y, X, features=12):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    model_scaler = StandardScaler()
    X_train_scaled = model_scaler.fit_transform(X_train)
    X_test_scaled = model_scaler.fit_transform(X_test)
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=features)
    rfe.fit(X_train_scaled, y_train)
    print(X.columns[rfe.support_])
    print(dict(zip(X.columns, rfe.ranking_)))
    print(rfe.ranking_)
    print(r2_score(y_test, rfe.predict(X_test_scaled)))
    print(mean_squared_error(y_test, rfe.predict(X_test_scaled)))


log_model(y_0, X_0)
rfe_review(y_0, X_0)


# Refined model independent variables
model_indp_var_new = ['model_constant', 'Initial_days', 'Age', 'Additional_charges', 'Complication_risk_Low',
                      'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
                      'Initial_admin_Observation Admission', 'Arthritis_Yes', 'Anxiety_Yes', 'Asthma_Yes',
                      'Stroke_Yes', 'Services_MRI', 'Services_CT Scan']
model_dep_var_new = 'ReAdmis_Yes'
X_1 = df_med[model_indp_var_new]
y_1 = df_med[model_dep_var_new]


log_model(y_1, X_1)


