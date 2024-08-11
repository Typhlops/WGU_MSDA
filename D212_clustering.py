import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans


df_med = pd.read_csv('data/D212-medical-files/medical_clean.csv')
original_df_med = pd.read_csv('data/D212-medical-files/medical_clean.csv')
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

# Saving cleaned data set
df_med.to_csv('medical_cleaned_clustering.csv')


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
        X_df.to_csv('X_scaled.csv')
    return X_df


# Adapted from Dr. Kesselly Kamara's “Evaluating and visualizing the model_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9fa8783e-d7d2-4b4d-b06e-b0ee01874bea
# Creates k-means cluster model using KMeans() from sklearn. Displays number of points in each cluster, prints
# centroid coordinates, and returns the scaled dataframe, k-means model, and the dataframe of the centroids.
def cluster_model(X, scaling, num_cluster, n_runs, rand_state):
    X_scaled = scale_data(X, scaling, True)
    kmodel = KMeans(n_clusters=num_cluster, n_init=n_runs, random_state=rand_state)
    kmodel.fit(X_scaled)

    evaluate_model = pd.Series(kmodel.labels_).value_counts()
    print(f"\nCounts by model cluster:\n{evaluate_model}")

    model_centroids = pd.DataFrame(kmodel.cluster_centers_, columns=X.columns)
    print(f"\nModel centroids:\n{model_centroids}")

    return X_scaled, kmodel, model_centroids


# Adapted from Dr. Kesselly Kamara's “Evaluating and visualizing the model_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9fa8783e-d7d2-4b4d-b06e-b0ee01874bea
# Creates a scatter plot for y vs x then creates a second scatter plot of y vs x using the clusters determined by the
# k-means model k_means_model. The points are colored and the centroid is labeled according to a legend provided.
def centroid_scatter(x, y, data_scaled, k_means_model, centroids, show_plain=True, show_discrete=True, df=df_med):
    scatter_size = 50
    p_palette = 'colorblind'
    p_alpha = 0.9
    p_size = scatter_size

    c_palette = p_palette
    c_size = 900
    c_marker = 'D'
    c_edge_color = 'black'

    text_align = 'center'
    text_h_align = text_align
    text_v_align = text_align
    text_size = 15
    text_weight = 'bold'
    text_color = 'white'

    if not show_discrete:
        # Operates from assumption that one-hot encoded variables are int32 while other numerical data is int64/float64.
        if (df[x].dtype == 'int32') or (df[y].dtype == 'int32'):
            print(f"Not producing scatter plot of {y} vs {x} as they are both one-hot encoded variables and the "
                  f"'show_discrete' option was set to False. Exiting.")
            return

    if show_plain:
        sns.scatterplot(data=df, x=x, y=y, s=scatter_size)
        plt.title(f"Scatter plot of {y} vs {x}")
        plt.show()

    plt.figure(figsize=(12, 10))
    ax_cs = sns.scatterplot(data=data_scaled, x=x, y=y, hue=k_means_model.labels_, palette=p_palette, alpha=p_alpha,
                            s=p_size, legend=True)
    ax_cs = sns.scatterplot(data=centroids, x=x, y=y, hue=centroids.index, palette=c_palette, s=c_size, marker=c_marker,
                            ec=c_edge_color, legend=False)
    for i in range(len(centroids)):
        plt.text(x=centroids[x][i], y=centroids[y][i], s=str(i), horizontalalignment=text_h_align,
                 verticalalignment=text_v_align, size=text_size, weight=text_weight, color=text_color)

    plt.title(f"Scatter plot of {y} vs {x} with colored clusters")
    plt.show()
    return


# Adapted from Dr. Kesselly Kamara's “Evaluating and visualizing the model_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9fa8783e-d7d2-4b4d-b06e-b0ee01874bea
# Creates a series of scatter plots using centroid_scatter on all possible pairs of variables in
# cols (intended to be the variables used in the provided k_means_model).
def centroid_graphs(cols, data_scaled, k_means_model, centroids, show_plain=False, show_discrete=True, df=df_med):
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            z_0 = cols[i]
            z_1 = cols[j]
            print(f"\nPlotting {z_1} vs {z_0}")
            centroid_scatter(z_0, z_1, data_scaled, k_means_model, centroids, show_plain, show_discrete, df)
    return


# Adapted from Dr. Kesselly Kamara's “Evaluating and visualizing the model_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9fa8783e-d7d2-4b4d-b06e-b0ee01874bea
# Calculates and outputs the inertia of the provided k_means_model. Then calculates the inertia for alternative
# instances of KMeans() with n_clusters = k for k in range(2, max_cluster). The inertia values are added to a series
# which is then used in a lineplot of model inertia vs n_clusters.
def wcss_eval(data_scaled, k_means_model, max_cluster=20, n_runs=50, rand_state=64):
    point_size = 200
    x_label = 'Optimal number of clusters (k)'
    y_label = 'Within Cluster Sum of Squares (WCSS)'
    ttl = 'WCSS vs number of clusters'

    k_inertia = k_means_model.inertia_
    print(f"\nModel inertia:\n{k_inertia}")

    wcss = []
    for k in range(2, max_cluster):
        model = KMeans(n_clusters=k, n_init=n_runs, random_state=rand_state)
        model.fit(data_scaled)
        wcss.append(model.inertia_)
    wcss_s = pd.Series(wcss, index=range(2, max_cluster))

    plt.figure(figsize=(12, 10))
    ax = sns.lineplot(x=wcss_s.index, y=wcss_s)
    ax = sns.scatterplot(x=wcss_s.index, y=wcss_s, s=point_size)
    ax = ax.set(xlabel=x_label, ylabel=y_label, title=ttl)
    plt.show()

    return k_inertia, wcss_s


# Adapted from Dr. Kesselly Kamara's “Evaluating and visualizing the model_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9fa8783e-d7d2-4b4d-b06e-b0ee01874bea
# Calculates and outputs the silhouette score of the provided k_means_model. Then calculates the silhouette score for
# alternative instances of KMeans() with n_clusters = k for k in range(2, max_cluster). The silhouette scores are added
# to a series which is then used in a lineplot of model silhouette score vs n_clusters.
def silhouette_eval(data_scaled, k_means_model, max_cluster=20, n_runs=50, rand_state=64):
    point_size = 200
    x_label = 'Optimal number of clusters (k)'
    y_label = 'Silhouette score average'
    ttl = 'Silhouette score vs number of clusters'

    k_silhouette_score = silhouette_score(data_scaled, k_means_model.labels_)
    print(f"\nModel silhouette score:\n{k_silhouette_score}")

    silhouette = []
    for k in range(2, max_cluster):
        model = KMeans(n_clusters=k, n_init=n_runs, random_state=rand_state)
        model.fit(data_scaled)
        silhouette.append(silhouette_score(data_scaled, model.labels_))

    silhouette_s = pd.Series(silhouette, index=range(2, max_cluster))

    plt.figure(figsize=(12, 10))
    ax = sns.lineplot(x=silhouette_s.index, y=silhouette_s)
    ax = sns.scatterplot(x=silhouette_s.index, y=silhouette_s, s=point_size)
    ax = ax.set(xlabel=x_label, ylabel=y_label, title=ttl)
    plt.show()

    return k_silhouette_score, silhouette_s


# Adapted from Dr. Kesselly Kamara's “Analyze and interpret K-means results_default” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3fe13831-fe4b-4c6b-a3eb-b0ee018754bc
# Adds a 'cluster' label column to the dataframe df (df_med by default). A dataframe df_cluster_agg is created which
# groups the observations by cluster label and provides the specified aggregates in aggr_dict. If plots is True,
# produces plots of the aggregated features by cluster label.
def cluster_agg(k_means_model, aggr_dict, plots=True, df=df_med):
    df['cluster'] = k_means_model.labels_.tolist()
    df_cluster_agg = df.groupby('cluster').agg(aggr_dict)
    print(f"\nClusters aggregated by {aggr_dict}:\n{df_cluster_agg}")
    print(f"\nAggregate values of the above specification across entire dataset:\n{pd.DataFrame([df.agg(aggr_dict)])}")

    if plots:
        point_size = 200
        x_label = 'Cluster number'
        for k, v in aggr_dict.items():
            y_label = k
            ttl = f"Value for {v} of {k} by cluster number"
            plt.figure(figsize=(12, 10))
            ax = sns.lineplot(data=df_cluster_agg, x='cluster', y=k)
            ax = sns.scatterplot(data=df_cluster_agg, x='cluster', y=k, s=point_size)
            ax = ax.set(xlabel=x_label, ylabel=y_label, title=ttl)
            plt.show()

    return df_cluster_agg


# Selecting initial features to form k-means clusters on
cluster_columns_0 = ['Initial_days', 'Age', 'Children', 'Income', 'Additional_charges', 'Population', 'ReAdmis_Yes',
                     'Complication_risk_Low', 'Services_MRI', 'Initial_admin_Observation Admission']
X_0 = df_med[cluster_columns_0]

# Saving subset of clean data for independent and dependent variables
X_0.to_csv('medical_transformed_clustering.csv')


# Unifies above functions to produce a k-means model and various evaluation metrics. display_plots=1 is default
# behavior to use centroid_graphs. Set display_plots=2 to display the various plots by cluster label that can be
# produced from cluster_agg.
def model_production(X, num_clusters, max_clusters, clust_cols, agg_dict, scaling='standard', num_runs=50, r_state=64,
                     display_plots=1, df=df_med):
    X_df, k_model, centroid = cluster_model(X, scaling, num_clusters, num_runs, r_state)
    if display_plots > 0:
        centroid_graphs(clust_cols, X_df, k_model, centroid, False, False, df)
    wcss_eval(X_df, k_model, max_clusters, num_runs, r_state)
    silhouette_eval(X_df, k_model, max_clusters, num_runs, r_state)
    cluster_agg(k_model, agg_dict, display_plots >= 2, df)
    return


aggr_dict = {'Initial_days': 'median', 'Age': 'median', 'Children': 'mean', 'Income': 'median', 'Population': 'median',
             'Additional_charges': 'median', 'Lat': 'median', 'Lng': 'median', 'Doc_visits': 'mean',
             'ReAdmis_Yes': 'mean', 'Complication_risk_Low': 'mean', 'Services_MRI': 'mean',
             'Initial_admin_Observation Admission': 'mean', 'Gender_Male': 'mean'}
model_production(X_0, 7, 15, cluster_columns_0, aggr_dict, 'standard', 50, 64, 0)


cluster_columns_1 = ['Initial_days', 'Age', 'Children', 'Income', 'Additional_charges', 'Population', 'Lat', 'Lng',
                     'VitD_levels', 'TotalCharge', 'vitD_supp', 'Full_meals_eaten', 'Doc_visits']
X_1 = df_med[cluster_columns_1]
X_1.to_csv('medical_transformed_clustering.csv')
model_production(X_1, 7, 15, cluster_columns_1, aggr_dict, 'standard', 50, 64, 0)


cluster_columns_2 = ['Initial_days', 'Age', 'Children', 'Income', 'Additional_charges', 'Population']
X_2 = df_med[cluster_columns_2]
X_2.to_csv('medical_transformed_clustering.csv')
model_production(X_2, 7, 15, cluster_columns_2, aggr_dict, 'standard', 50, 64, 2)

