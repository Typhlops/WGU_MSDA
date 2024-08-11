import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

df_med = pd.read_csv('C:/Users/Joe/Desktop/Files/Programming/WGU MS/Python_projects/data/D212-medical-files/medical_clean.csv')
#original_df_med = pd.read_csv('data/D212-medical-files/medical_clean.csv')
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


# Scales independent variables in X with a StandardScaler, MinMaxScaler, RobustScaler, or nothing. When save is True,
# the scaled data is saved to a csv file. Returns the scaled dataframe X_df.
def scale_data(X, scl='standard', save=True):
    if scl == 'standard':
        scaler = StandardScaler()
    elif scl == 'minmax':
        scaler = MinMaxScaler()
    elif scl == 'robust':
        scaler = RobustScaler()
    elif scl == 'none':
        X_df = X
    else:
        print(f"Invalid choice of scaler {scl}. Exiting.")
        return

    if scl != 'none':
        # X_numerical more accurately means columns that weren't one-hot encoded from categorical columns.
        # The function one_hot_encoder outputs int32 0s and 1s. With this dataset, numerical columns are int64/float64.
        X_numerical = X.select_dtypes({'float64', 'int64'})
        X_encoded = X.select_dtypes('int32')
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        df_nmr = pd.DataFrame(X_numerical_scaled, columns=X_numerical.columns)
        X_df = pd.concat([df_nmr, X_encoded], axis=1)

    if save:
        X_df.to_csv('X_pca_scaled.csv')
    return X_df


# Setting PCA columns
pca_columns_med = ['Initial_days', 'Age', 'Children', 'Income', 'Additional_charges', 'Population', 'Lat', 'Lng',
                   'VitD_levels', 'TotalCharge', 'vitD_supp', 'Full_meals_eaten', 'Doc_visits']
X_pca = scale_data(df_med[pca_columns_med], 'none', False)


# Adapted from WGU D206 Data Cleaning "Welcome to Getting Started With Principal Component Analysis". Accessed 2024.
# Performs PCA and produces scree plot to determine optimal number of features
def pca_procedure(data_scaled, num_components='max', r_state=72, plots=True):
    if num_components == 'max':
        pca_col_count = data_scaled.shape[1]
    elif type(num_components) is int:
        if num_components > 0:
            pca_col_count = num_components
        else:
            print("Error: invalid entry for num_components. Must be 'max' or a valid positive integer.")
            return
    else:
        print("Error: invalid entry for num_components. Must be 'max' or a valid positive integer.")
        return
    pca = PCA(n_components=pca_col_count, random_state=r_state)
    pca.fit(data_scaled)
    pca_col_names = ['PC' + f'{i + 1}' for i in range(pca_col_count)]
    df_pca = pd.DataFrame(pca.transform(data_scaled), columns=pca_col_names)

    pd.set_option('display.max_columns', 5)
    print("\nPCA dataframe:\n")
    print(df_pca)
    pd.set_option('display.max_columns', None)

    pca_loadings = pd.DataFrame(pca.components_.T, columns=pca_col_names, index=data_scaled.columns)
    print("\nPCA loadings:\n")
    print(pca_loadings)
    df_exp_var = pd.DataFrame({'Explained_variance': pca.explained_variance_}, index=pca_col_names)
    print("\nPCA explained variance:\n")
    print(df_exp_var)
    df_exp_var_ratio = pd.DataFrame({'Explained_variance_ratio': pca.explained_variance_ratio_}, index=pca_col_names)
    print("\nPCA explained variance ratio:\n")
    print(df_exp_var_ratio)
    df_cum_exp_var_ratio = pd.DataFrame({'Cumulative explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_)}, index=pca_col_names)
    print("\nPCA cumulative explained variance ratio:\n")
    print(df_cum_exp_var_ratio)

    if plots:
        ax0 = sns.lineplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=pca.explained_variance_ratio_)
        ax0 = sns.scatterplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=pca.explained_variance_ratio_)
        ax0 = ax0.set(xlabel='PCA component number', ylabel='Explained variance ratio', title='Explained variance ratio by PCA component number')
        plt.show()
        ax1 = sns.lineplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=np.cumsum(pca.explained_variance_ratio_))
        ax1 = sns.scatterplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=np.cumsum(pca.explained_variance_ratio_))
        ax1 = ax1.set(xlabel='PCA component number', ylabel='Cumulative explained variance ratio', title='Cumulative explained variance ratio by PCA component number')
        plt.show()

    pca_cov_matrix = np.dot(data_scaled.T, data_scaled) / data_scaled.shape[0]
    pca_eigenvalues = [np.dot(eigenvector.T, np.dot(pca_cov_matrix, eigenvector)) for eigenvector in pca.components_]
    if plots:
        ax2 = sns.lineplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=pca_eigenvalues)
        ax2 = sns.scatterplot(x=range(1, len(pca.explained_variance_ratio_)+1), y=pca_eigenvalues)
        ax2 = ax2.set(xlabel='Number of components', ylabel='PCA eigenvalue', title='PCA eigenvalue by component number')
        sns.lineplot(x=range(3,10), y=-1, color='red')
        #plt.axhline(y=1, color='blue')
        plt.show()


pca_procedure(X_pca, 10)


