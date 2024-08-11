import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import missingno as msno
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)

df_a = pd.read_csv('all_Spotify_artist_info.csv')
df_t = pd.read_csv('all_Spotify_track_info.csv')

df_a = df_a.drop(columns=['Unnamed: 0'])
df_t = df_t.drop(columns=['Unnamed: 0'])


# Adding custom 'range' column (max - min)
def add_range_col(df):
    val_, valence_, min_, max_, range_ = '_val', 'valence', '_min', '_max', '_range'
    all_df_cols_nmr = list(df.select_dtypes(exclude='object').columns)
    prefixes = {s.removesuffix(val_) for s in all_df_cols_nmr if (val_ in s and valence_ not in s) or
                ((valence_ + val_) in s)}

    for prefix in prefixes:
        if (prefix + max_) in all_df_cols_nmr:
            df[(prefix + range_)] = df.apply(lambda row: row[(prefix + max_)] - row[(prefix + min_)], axis=1)
    return


add_range_col(df_a)
add_range_col(df_t)


# Dataframe columns and various subsets for convenient use
all_a_columns = ['featured_count', 'dates', 'ids', 'names', 'genres', 'first_release', 'last_release', 'num_releases',
                 'num_tracks', 'playlists_found', 'feat_track_ids', 'tracks', 'ambient', 'asian', 'christian',
                 'classical_instrumental', 'contemporary', 'country', 'electronic', 'euro', 'folk', 'hip-hop',
                 'indie_alternative', 'jazz', 'latin', 'metal', 'pop', 'reggae_soul', 'rock', 'monthly_listeners_mean',
                 'monthly_listeners_median', 'monthly_listeners_std_dev', 'monthly_listeners_min',
                 'monthly_listeners_max', 'monthly_listeners_first', 'monthly_listeners_last', 'monthly_listeners_val',
                 'popularity_mean', 'popularity_median', 'popularity_std_dev', 'popularity_min', 'popularity_max',
                 'popularity_first', 'popularity_last', 'popularity_val', 'followers_mean', 'followers_median',
                 'followers_std_dev', 'followers_min', 'followers_max', 'followers_first', 'followers_last',
                 'followers_val', 'log_monthly_listeners_mean', 'log_monthly_listeners_median',
                 'log_monthly_listeners_std_dev', 'log_monthly_listeners_min', 'log_monthly_listeners_max',
                 'log_monthly_listeners_first', 'log_monthly_listeners_last', 'log_monthly_listeners_val',
                 'track_popularity_val', 'track_popularity_mean', 'track_popularity_median', 'track_popularity_std_dev',
                 'track_popularity_min', 'track_popularity_max', 'track_release_date_val', 'track_release_date_mean',
                 'track_release_date_median', 'track_release_date_std_dev', 'track_release_date_min',
                 'track_release_date_max', 'track_duration_ms_val', 'track_duration_ms_mean',
                 'track_duration_ms_median', 'track_duration_ms_std_dev', 'track_duration_ms_min',
                 'track_duration_ms_max', 'track_acousticness_val', 'track_acousticness_mean',
                 'track_acousticness_median', 'track_acousticness_std_dev', 'track_acousticness_min',
                 'track_acousticness_max', 'track_danceability_val', 'track_danceability_mean',
                 'track_danceability_median', 'track_danceability_std_dev', 'track_danceability_min',
                 'track_danceability_max', 'track_energy_val', 'track_energy_mean', 'track_energy_median',
                 'track_energy_std_dev', 'track_energy_min', 'track_energy_max', 'track_instrumentalness_val',
                 'track_instrumentalness_mean', 'track_instrumentalness_median', 'track_instrumentalness_std_dev',
                 'track_instrumentalness_min', 'track_instrumentalness_max', 'track_liveness_val',
                 'track_liveness_mean', 'track_liveness_median', 'track_liveness_std_dev', 'track_liveness_min',
                 'track_liveness_max', 'track_loudness_val', 'track_loudness_mean', 'track_loudness_median',
                 'track_loudness_std_dev', 'track_loudness_min', 'track_loudness_max', 'track_speechiness_val',
                 'track_speechiness_mean', 'track_speechiness_median', 'track_speechiness_std_dev',
                 'track_speechiness_min', 'track_speechiness_max', 'track_tempo_val', 'track_tempo_mean',
                 'track_tempo_median', 'track_tempo_std_dev', 'track_tempo_min', 'track_tempo_max', 'track_valence_val',
                 'track_valence_mean', 'track_valence_median', 'track_valence_std_dev', 'track_valence_min',
                 'track_valence_max', 'track_musicalkey_val', 'track_musicalkey_mode', 'track_musicalmode_val',
                 'track_musicalmode_mode', 'track_time_signature_val', 'track_time_signature_mode',
                 'track_duration_ms_range', 'popularity_range', 'log_monthly_listeners_range',
                 'track_instrumentalness_range', 'track_loudness_range', 'track_energy_range', 'track_tempo_range',
                 'followers_range', 'track_release_date_range', 'track_liveness_range', 'track_speechiness_range',
                 'track_valence_range', 'track_acousticness_range', 'track_danceability_range',
                 'monthly_listeners_range', 'track_popularity_range']

all_t_columns = ['featured', 'ids', 'names', 'popularity', 'markets', 'artists', 'release_date', 'duration_ms',
                 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness',
                 'tempo', 'valence', 'musicalkey', 'musicalmode', 'time_signature', 'count', 'dates', 'playlists_found',
                 'release_year', 'artist_featured_count', 'artist_dates', 'artist_ids', 'artist_names',
                 'artist_first_release', 'artist_last_release', 'artist_num_releases', 'artist_num_tracks',
                 'artist_playlists_found', 'artist_feat_track_ids', 'artist_tracks', 'artist_ambient', 'artist_asian',
                 'artist_christian', 'artist_classical_instrumental', 'artist_contemporary', 'artist_country',
                 'artist_electronic', 'artist_euro', 'artist_folk', 'artist_hip-hop', 'artist_indie_alternative',
                 'artist_jazz', 'artist_latin', 'artist_metal', 'artist_pop', 'artist_reggae_soul', 'artist_rock',
                 'artist_monthly_listeners_mean', 'artist_monthly_listeners_median', 'artist_monthly_listeners_std_dev',
                 'artist_monthly_listeners_min', 'artist_monthly_listeners_max', 'artist_monthly_listeners_val',
                 'artist_popularity_mean', 'artist_popularity_median', 'artist_popularity_std_dev',
                 'artist_popularity_min', 'artist_popularity_max', 'artist_popularity_val', 'artist_followers_mean',
                 'artist_followers_median', 'artist_followers_std_dev', 'artist_followers_min', 'artist_followers_max',
                 'artist_followers_val', 'artist_log_monthly_listeners_mean', 'artist_log_monthly_listeners_median',
                 'artist_log_monthly_listeners_std_dev', 'artist_log_monthly_listeners_min',
                 'artist_log_monthly_listeners_max', 'artist_log_monthly_listeners_val',
                 'artist_monthly_listeners_range', 'artist_log_monthly_listeners_range', 'artist_followers_range',
                 'artist_popularity_range']

all_a_cols_nmr = list(df_a.select_dtypes(exclude='object').columns)
all_a_cols_str = list(df_a.select_dtypes(include='object').columns)
a_mean_cols = [s for s in all_a_columns if '_mean' in s]
a_median_cols = [s for s in all_a_columns if '_median' in s]
a_std_cols = [s for s in all_a_columns if '_std' in s]
a_min_cols = [s for s in all_a_columns if '_min' in s]
a_max_cols = [s for s in all_a_columns if '_max' in s]
a_mode_cols = [s for s in all_a_columns if '_mode' in s]
a_first_cols = [s for s in all_a_columns if '_first' in s]
a_last_cols = [s for s in all_a_columns if '_last' in s]
a_val_cols = [s for s in all_a_columns if ('_val' in s and '_valence' not in s) or (s == 'track_valence_val')]
a_range_cols = [s for s in all_a_columns if '_range' in s]
a_artist_cols = [s for s in all_a_columns if 'track_' not in s or 'feat_track' in s]
a_track_cols = [s for s in all_a_columns if 'track_' in s and 'feat_track' not in s]
a_genre_cols = ['ambient', 'asian', 'christian', 'classical_instrumental', 'contemporary', 'country', 'electronic',
                'euro', 'folk', 'hip-hop', 'indie_alternative', 'jazz', 'latin', 'metal', 'pop', 'reggae_soul', 'rock']

all_t_cols_nmr = list(df_t.select_dtypes(exclude='object').columns)
all_t_cols_str = list(df_t.select_dtypes(include='object').columns)
t_mean_cols = [s for s in all_t_columns if '_mean' in s]
t_median_cols = [s for s in all_t_columns if '_median' in s]
t_std_cols = [s for s in all_t_columns if '_std' in s]
t_min_cols = [s for s in all_t_columns if '_min' in s]
t_max_cols = [s for s in all_t_columns if '_max' in s]
t_val_cols = [s for s in all_t_columns if '_val' in s]
t_range_cols = [s for s in all_t_columns if '_range' in s]
t_artist_cols = [s for s in all_t_columns if 'artist_' in s]
t_track_cols = [s for s in all_t_columns if 'artist_' not in s]
t_genre_cols = ['artist_' + g for g in a_genre_cols]

# Setting top level seed and state values
val_seed = 32
val_state = 74

# Target variables
target_a_log_m = 'log_monthly_listeners_val'
target_a_monthly = 'monthly_listeners_val'
target_a_pop = 'popularity_val'
target_t_art_log_m = 'artist_log_monthly_listeners_val'
target_t_art_m = 'artist_monthly_listeners_val'
target_t_pop = 'popularity'

# Random forest grid search parameters
param_dict_small = {"n_estimators": [500],
                    "max_features": [6, 8, 10],
                    "max_depth": [None],
                    "min_samples_split": [6, 8],
                    "min_samples_leaf": [6, 10]}

param_dict = {"n_estimators": [350, 500],
              "max_features": [6, 8, 10],
              "max_depth": [8, None],
              "min_samples_split": [4, 6, 8],
              "min_samples_leaf": [4, 6, 8]}

# Subset of columns for data exploration and pipelines
reduced_a_columns = ['featured_count', 'first_release', 'last_release', 'num_releases', 'num_tracks', 'ambient',
                     'asian', 'christian', 'classical_instrumental', 'contemporary', 'country', 'electronic', 'euro',
                     'folk', 'hip-hop', 'indie_alternative', 'jazz', 'latin', 'metal', 'pop', 'reggae_soul', 'rock',
                     'monthly_listeners_val', 'popularity_val', 'followers_val', 'log_monthly_listeners_val',
                     'track_popularity_val', 'track_release_date_val', 'track_duration_ms_val',
                     'track_acousticness_val', 'track_danceability_val', 'track_energy_val',
                     'track_instrumentalness_val', 'track_liveness_val', 'track_loudness_val', 'track_speechiness_val',
                     'track_tempo_val', 'track_valence_val', 'track_musicalkey_val', 'track_musicalmode_val',
                     'track_time_signature_val', 'monthly_listeners_std_dev', 'popularity_std_dev', 'followers_std_dev',
                     'log_monthly_listeners_std_dev', 'track_popularity_std_dev', 'track_release_date_std_dev',
                     'track_tempo_std_dev']

reduced_t_columns = ['featured', 'popularity', 'duration_ms', 'acousticness', 'danceability', 'energy',
                     'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'musicalkey',
                     'musicalmode', 'time_signature', 'count', 'release_year', 'artist_featured_count',
                     'artist_first_release', 'artist_num_releases', 'artist_num_tracks', 'artist_ambient',
                     'artist_asian', 'artist_classical_instrumental', 'artist_electronic', 'artist_euro',
                     'artist_hip-hop', 'artist_metal', 'artist_pop', 'artist_rock', 'artist_monthly_listeners_std_dev',
                     'artist_monthly_listeners_val', 'artist_popularity_std_dev', 'artist_popularity_val',
                     'artist_followers_std_dev', 'artist_followers_val', 'artist_log_monthly_listeners_std_dev',
                     'artist_log_monthly_listeners_val']

# Subset of columns to use for recursive feature elimination
rfe_cols_a = ['featured_count', 'first_release', 'last_release', 'num_releases', 'num_tracks', 'monthly_listeners_mean',
              'monthly_listeners_std_dev', 'monthly_listeners_min', 'monthly_listeners_max', 'monthly_listeners_val',
              'popularity_mean', 'popularity_std_dev', 'popularity_min', 'popularity_max', 'popularity_val',
              'followers_mean', 'followers_std_dev', 'followers_min', 'followers_max', 'followers_val',
              'log_monthly_listeners_mean', 'log_monthly_listeners_std_dev', 'log_monthly_listeners_min',
              'log_monthly_listeners_max', 'log_monthly_listeners_val', 'track_popularity_val', 'track_popularity_mean',
              'track_popularity_std_dev', 'track_popularity_min', 'track_popularity_max', 'track_release_date_val',
              'track_release_date_mean', 'track_release_date_std_dev', 'track_release_date_min',
              'track_release_date_max', 'track_duration_ms_val', 'track_duration_ms_mean', 'track_duration_ms_std_dev',
              'track_duration_ms_min', 'track_duration_ms_max', 'track_acousticness_val', 'track_acousticness_mean',
              'track_acousticness_std_dev', 'track_acousticness_min', 'track_acousticness_max',
              'track_danceability_val', 'track_danceability_mean', 'track_danceability_std_dev',
              'track_danceability_min', 'track_danceability_max', 'track_energy_val', 'track_energy_mean',
              'track_energy_std_dev', 'track_energy_min', 'track_energy_max', 'track_instrumentalness_val',
              'track_instrumentalness_mean', 'track_instrumentalness_std_dev', 'track_instrumentalness_min',
              'track_instrumentalness_max', 'track_liveness_val', 'track_liveness_mean', 'track_liveness_std_dev',
              'track_liveness_min', 'track_liveness_max', 'track_loudness_val', 'track_loudness_mean',
              'track_loudness_std_dev', 'track_loudness_min', 'track_loudness_max', 'track_speechiness_val',
              'track_speechiness_mean', 'track_speechiness_std_dev', 'track_speechiness_min', 'track_speechiness_max',
              'track_tempo_val', 'track_tempo_mean', 'track_tempo_std_dev', 'track_tempo_min', 'track_tempo_max',
              'track_valence_val', 'track_valence_mean', 'track_valence_std_dev', 'track_valence_min',
              'track_valence_max', 'track_musicalkey_val', 'track_musicalmode_val', 'track_time_signature_val',
              'track_duration_ms_range', 'popularity_range', 'log_monthly_listeners_range',
              'track_instrumentalness_range', 'track_loudness_range', 'track_energy_range', 'track_tempo_range',
              'followers_range', 'track_release_date_range', 'track_liveness_range', 'track_speechiness_range',
              'track_valence_range', 'track_acousticness_range', 'track_danceability_range', 'monthly_listeners_range',
              'track_popularity_range', 'ambient', 'asian', 'christian', 'classical_instrumental', 'contemporary',
              'country', 'electronic', 'euro', 'folk', 'hip-hop', 'indie_alternative', 'jazz', 'latin', 'metal', 'pop',
              'reggae_soul', 'rock']

rfe_cols_t = ['featured', 'popularity', 'markets', 'duration_ms', 'acousticness', 'danceability', 'energy',
              'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'musicalkey',
              'musicalmode', 'time_signature', 'count', 'release_year', 'artist_featured_count', 'artist_first_release',
              'artist_last_release', 'artist_num_releases', 'artist_num_tracks', 'artist_monthly_listeners_mean',
              'artist_monthly_listeners_std_dev', 'artist_monthly_listeners_min', 'artist_monthly_listeners_max',
              'artist_monthly_listeners_val', 'artist_popularity_mean', 'artist_popularity_std_dev',
              'artist_popularity_min', 'artist_popularity_max', 'artist_popularity_val', 'artist_followers_mean',
              'artist_followers_std_dev', 'artist_followers_min', 'artist_followers_max', 'artist_followers_val',
              'artist_log_monthly_listeners_mean', 'artist_log_monthly_listeners_std_dev',
              'artist_log_monthly_listeners_min', 'artist_log_monthly_listeners_max',
              'artist_log_monthly_listeners_val', 'artist_monthly_listeners_range',
              'artist_log_monthly_listeners_range', 'artist_followers_range', 'artist_popularity_range',
              'artist_ambient', 'artist_asian', 'artist_christian', 'artist_classical_instrumental',
              'artist_contemporary', 'artist_country', 'artist_electronic', 'artist_euro', 'artist_folk',
              'artist_hip-hop', 'artist_indie_alternative', 'artist_jazz', 'artist_latin', 'artist_metal', 'artist_pop',
              'artist_reggae_soul', 'artist_rock']





# Dataframe .info() and searching for null values and duplicates
def info_print(df):
    print("Dataframe info:")
    df.info()
    print(f"\nInspecting number of nulls in columns:\n{df.isna().sum()[df.isna().sum() > 0]}\n")
    print(f"Looking for duplicates: {df.duplicated().sum()}\n")
    return


# Dataframe description and value counts
def inspect_data(columns, df):
    for col in columns:
        if (df[col].dtype == 'int64') or (df[col].dtype == 'float64'):
            print(f"\nNumber of unique values: {len(df[col].unique())}")
            print(df[col].describe())
        else:
            print(df[col].describe())
            print(df[col].value_counts(), "\n")
    return


# Creates plots across variables in (numeric) 'columns'. Mode can be histogram, pair plot, or scatter plot.
def plots_by_column(columns, df, mode='hist', y_scatter='popularity_val', save=False, display=True):
    suffix_lim = 3
    if mode == 'hist':
        for coln in columns:
            nonnull_df = df.loc[df[coln].notna()]
            plt.hist(nonnull_df[coln])
            plt.xlabel(coln)
            if save:
                plt.savefig(f"{mode}_{coln}.png")
            if display:
                plt.show()
    elif mode == 'pairplot':
        sns.pairplot(df[columns], diag_kind='hist')
        if save:
            suffix = ("-".join(columns[:suffix_lim]) + '-others') if len(columns) > suffix_lim else "-".join(columns)
            plt.savefig(f"{mode}_{suffix}.png")
        if display:
            plt.show()
    elif mode == 'scatter':
        for coln in columns:
            plt.scatter(x=df[coln], y=df[y_scatter])
            plt.xlabel(coln)
            plt.ylabel(y_scatter)
            if save:
                plt.savefig(f"{mode}_{y_scatter}-vs-{coln}.png")
            if display:
                plt.show()
            if save or display:
                plt.close()
    else:
        print(f"Invalid mode selection")
    return


# missingno matrix, dendrogram, and heatmap
def msno_plots(columns, df):
    df_numerical = df.select_dtypes(exclude='object')
    msno.matrix(df[columns], labels=True)
    plt.show()
    msno.dendrogram(df[columns])
    plt.show()
    msno.heatmap(df_numerical, labels=True, cmap='RdBu', fontsize=12, vmin=-0.1, vmax=0.1)
    plt.show()
    return


# Searches for outliers by IQR and z-scores (defaults to |z| > 3.0) with optional z-score histogram plot
def outlier_search(columns, df, plots=True, z_bound=3.0):
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
            df_outliers[col_zscore] = stats.zscore(df[column].dropna())
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


# Prints correlation matrix values restricted to a specific variable ('column') sorted in descending order of their
# absolute value. Optional threshold parameter 'thresh' defaults to -1.0 and restricts correlated variables to
# those above it.
def corr_search(column, df, thresh=-1.0):
    sorted_corr = df.select_dtypes(exclude=['object']).corr()[column].sort_values(ascending=False, key=lambda x: abs(x))
    print(sorted_corr[abs(sorted_corr) >= thresh], "\n")
    return


# Convenience function for data exploration
def extensive_exploration(a_cols_reduced, t_cols_reduced, df_art=df_a, df_track=df_t, show_plots=False, save=False):
    df_reduced_a = df_art[a_cols_reduced]
    df_reduced_t = df_track[t_cols_reduced]
    reduced_a_cols_nmr = list(df_reduced_a.select_dtypes(exclude='object').columns)
    reduced_t_cols_nmr = list(df_reduced_t.select_dtypes(exclude='object').columns)

    with pd.option_context('display.max_rows', None):
        corr_search(target_a_pop, df_reduced_a)
        corr_search(target_a_monthly, df_reduced_a)
        corr_search(target_a_log_m, df_reduced_a)
        corr_search(target_t_pop, df_reduced_t)
        corr_search(target_t_art_m, df_reduced_t)
        corr_search(target_t_art_log_m, df_reduced_t)
    inspect_data(df_reduced_a.columns, df_reduced_a)
    inspect_data(df_reduced_t.columns, df_reduced_t)
    outlier_search(reduced_a_cols_nmr, df_reduced_a, show_plots)
    outlier_search(reduced_t_cols_nmr, df_reduced_t, show_plots)
    if show_plots or save:
        plots_by_column(reduced_a_cols_nmr, df_reduced_a, 'scatter', target_a_pop, save, show_plots)
        plots_by_column(reduced_a_cols_nmr, df_reduced_a, 'scatter', target_a_monthly, save, show_plots)
        plots_by_column(reduced_a_cols_nmr, df_reduced_a, 'scatter', target_a_log_m, save, show_plots)
        plots_by_column(reduced_t_cols_nmr, df_reduced_t, 'scatter', target_t_pop, save, show_plots)
        plots_by_column(reduced_t_cols_nmr, df_reduced_t, 'scatter', target_t_art_m, save, show_plots)
        plots_by_column(reduced_t_cols_nmr, df_reduced_t, 'scatter', target_t_art_log_m, save, show_plots)
    return


# If the input feature_list contains invalid variables, they're removed and the updated list is returned.
# Specifically, it removes the target variable if present; any related statistical variables to the target (e.g. if
# target is 'popularity_val' then 'popularity_mean' etc. are removed); and if the target is predicting one of the
# 'monthly_listeners' ('log_monthly_listeners') statistical terms then all terms containing 'monthly_listeners' are
# removed (if present).
def valid_features(feature_list, target):
    popularity_ = 'popularity'
    monthly_listeners_ = 'monthly_listeners'
    std_dev_ = 'std_dev'
    stat_terms = ['mean', 'median', 'std_dev', 'min', 'max', 'mode', 'first', 'last', 'val', 'range']

    if std_dev_ in target:
        target_root = target[:target.rfind('_', 0, target.rfind('_'))]
    elif '_' in target:
        target_root = target[:target.rfind('_')]
    else:
        target_root = ''

    if target_root:
        related_to_target_terms = [(target_root + '_' + term) for term in stat_terms]
        feature_list = [s for s in feature_list if s not in related_to_target_terms]

    if monthly_listeners_ in target:
        feature_list = [s for s in feature_list if monthly_listeners_ not in s]

    if popularity_ in target:
        feature_list = [s for s in feature_list if popularity_ not in s]

    if target in feature_list:
        feature_list.remove(target)

    return feature_list


# Creates y and X sub-dataframes from df with target (intended to be target_var) and predictors (intended to be
# top_features (the output of valid_features from input top_features_0)). top_features are determined by the output
# of feature_selection and hyper_search (found further below). For hyper_feature (found further below) to operate as
# expected, top_features should be in ascending order of p-values as determined by an F-test (f_regression). X_00
# restricts df to columns including valid_types (np.number) and removes the target column. If validate=True,
# it runs valid_features to ensure the independent variables are appropriate for the selected target (see
# valid_features documentation for more information), potentially restricting X_00 further. X_11 restricts df to
# predictors (or the output of valid features on those predictors if validate=True). The sub-dataframe on (valid)
# predictors and the target columns will be saved to save_name (if not an empty string).
def create_y_X_frames(predictors, target, df, validate=True, save_name=''):
    valid_types = np.number

    if validate:
        valid_cols_00 = valid_features(list(df.select_dtypes(include=valid_types).columns), target)
        valid_predictors = [p for p in predictors if p in valid_cols_00]
    else:
        valid_cols_00 = list(df.select_dtypes(include=valid_types).columns)
        valid_predictors = predictors

    y_00 = df[target]
    X_00 = df[valid_cols_00]
    X_11 = df[valid_predictors]

    # Saving subset of clean data for independent and dependent variables
    if save_name:
        Z_00 = df[valid_predictors + [target]]
        Z_00.to_csv(save_name)
    return y_00, X_00, X_11


# Scales independent variables in X with a StandardScaler, MinMaxScaler, RobustScaler, or nothing. If split is True,
# the X and y inputs are split into training and test data that are saved to .csv files when save is True. Returns the
# scaled dataframe X_df (and the train/test split if applicable).
def scale_split(y, X, scl='standard', split=True, save=True, split_size=0.2, seed=val_seed):
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
    vif_data = pd.DataFrame({'Feature': X.columns,
                             'VIF': [variance_inflation_factor(X.values, i) for
                                     i in range(len(X.columns))]}).sort_values(by='VIF', ascending=False)
    with pd.option_context('display.max_rows', None):
        print("\n", vif_data)


# Uses SelectKBest to find the best features sorted by p-values (from an F-test). Uses alpha to restrict the sorted
# features to those with a p-value <= alpha. Returns a dataframe of the sorted features and list of columns with
# p-value below alpha.
def feature_selection(y, X, df, k_num='all', alpha=0.05, output=True):
    best_feat = SelectKBest(f_regression, k=k_num)
    best_feat.fit(X, y)
    df_features = pd.DataFrame({'Score': best_feat.scores_, 'p-value': best_feat.pvalues_},
                               index=best_feat.feature_names_in_).sort_values(by=['p-value', 'Score'],
                                                                              ascending=[True, False])
    refine_max_cols = k_num if isinstance(k_num, int) and k_num < len(df_features) else None
    df_refine = df_features[df_features['p-value'] <= alpha].iloc[:refine_max_cols]
    if output:
        with pd.option_context('display.max_rows', None):
            print(df_features)
        if isinstance(k_num, int):
            print(f"\nVIF for top {k_num} features with p-values <= {alpha}:")
        else:
            print(f"\nVIF for features with p-values <= {alpha}:")
        vif_print(df[df_refine.index])
    return df_features, df_refine.index


# Random forest regressor for target y with independent variables X
def predictor_rf(y, X, n_est=100, m_feat=4, m_dep=None, m_spl=2, m_leaf=2, split_frc=0.2, seed=val_seed,
                 state=val_state, save=False, output=True, save_plots=False, display_plots=True, cross_val=True):
    rel_percentile_low = 1
    rel_percentile_high = 99
    fig_dims = (18, 15)

    target_label = y.name
    plot_save_name = f"residuals_error_QQ_{target_label}.png"

    X_df, X_train, X_test, y_train, y_test = scale_split(y, X, 'none', True, save, split_frc, seed)

    rfr = RandomForestRegressor(n_estimators=n_est, max_features=m_feat, max_depth=m_dep,
                                min_samples_split=m_spl, min_samples_leaf=m_leaf, random_state=state)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    y_pred_train = rfr.predict(X_train)
    rsquare = r2_score(y_test, y_pred)
    rsquare_train = r2_score(y_train, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse = np.sqrt(mse)
    rmse_train = np.sqrt(mse_train)
    mae = mean_absolute_error(y_test, y_pred)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_ratio = rmse / rmse_train
    df_feat_import = pd.DataFrame({'feature_importance': rfr.feature_importances_},
                                  index=X_df.columns).sort_values(by='feature_importance', ascending=False)
    ordered_features = list(df_feat_import.index)

    if output:
        print(f"\nR^2: {rsquare}")
        print(f"\nR^2 on train: {rsquare_train}")
        print(f"\nMSE: {mse}")
        print(f"\nMSE on train: {mse_train}")
        print(f"\nRMSE: {rmse}")
        print(f"\nRMSE on train: {rmse_train}")
        print(f"\nMAE: {mae}")
        print(f"\nMAE on train: {mae_train}")
        print(f"\nRMSE test to train ratio: {rmse_ratio}")
        with pd.option_context('display.max_rows', None):
            print(f"\nFeature importances:\n{df_feat_import}")
        print(f"\nList of features in descending order of importance:\n{ordered_features}")

    if cross_val:
        print(f"\nCross validation scores: {cross_val_score(rfr, X_train, y_train, cv=5)}")

    if save_plots or display_plots:
        plot_fig = plt.figure(figsize=fig_dims)

        ax_resid = plot_fig.add_subplot(221)
        y_pred_all = rfr.predict(X)
        residuals = y - y_pred_all
        sns.scatterplot(x=y_pred_all, y=residuals, ax=ax_resid)
        plt.xlabel(f'Predicted value of {target_label}')
        plt.ylabel('Residuals')
        plt.title(f"Residuals vs predicted values of {target_label}")

        ax_relerr = plot_fig.add_subplot(222)
        relative_error = 1 - y_pred_all[y != 0]/y[y != 0]
        sns.scatterplot(x=y_pred_all[y != 0], y=relative_error, ax=ax_relerr)
        plt.ylim(np.percentile(relative_error, rel_percentile_low), np.percentile(relative_error, rel_percentile_high))
        plt.xlabel(f"Predicted value of {target_label}")
        plt.ylabel('Relative error')
        plt.title(f"Relative error vs predicted values\n"
                  f"(for percentiles {rel_percentile_low} to {rel_percentile_high} when {target_label} > 0)")

        ax_qq = plot_fig.add_subplot(223)
        sm.qqplot(residuals, ax=ax_qq)
        plt.title(f"QQ plot of residuals for {target_label}")

        plt.tight_layout()
        if save_plots:
            plt.savefig(plot_save_name)
        if display_plots:
            plt.show()

    return rfr, rsquare, mse, mae, rmse_ratio, ordered_features


# Adaptation of GridSearchCV hyperparametric tuning for the random forest regressor using the param_dict above as the
# intended param_grid dictionary. Prints optimal parameters and model statistics such as MSE and R^2. Returns optimal
# parameters, R^2, and MSE.
def hyper_param(y, X, parameters, scl='none', save=False, split_frc=0.2,
                seed=val_seed, state=val_state, verbosity=4):
    val_cv = 5
    X_df, X_train, X_test, y_train, y_test = scale_split(y, X, scl, True, save, split_frc, seed)
    rf_start = RandomForestRegressor(random_state=state)

    rf_grid = GridSearchCV(estimator=rf_start, param_grid=parameters, scoring='neg_mean_squared_error',
                           cv=val_cv, n_jobs=-1, verbose=verbosity)
    rf_grid.fit(X_train, y_train)

    opt_params = rf_grid.best_params_
    opt_rf = rf_grid.best_estimator_
    y_pred = opt_rf.predict(X_test)
    y_pred_train = opt_rf.predict(X_train)
    rsquare = r2_score(y_test, y_pred)
    rsquare_train = r2_score(y_train, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse = np.sqrt(mse)
    rmse_train = np.sqrt(mse_train)
    mae = mean_absolute_error(y_test, y_pred)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_ratio = rmse / rmse_train
    # df_results = pd.DataFrame(rf_grid.cv_results_)
    print(f"\nBest parameters: {opt_params}")
    print(f"\nR^2: {rsquare}")
    print(f"\nR^2 on train: {rsquare_train}")
    print(f"\nMSE: {mse}")
    print(f"\nMSE on train: {mse_train}")
    print(f"\nRMSE: {rmse}")
    print(f"\nRMSE on train: {rmse_train}")
    print(f"\nMAE: {mae}")
    print(f"\nMAE on train: {mae_train}")
    print(f"\nRMSE test to train ratio: {rmse_ratio}\n")
    return opt_params, rsquare, mse


# Hyperparametric tuning looping over which features are used as predictor variables in the random forest regressor.
# X_columns is intended to be top_features (output of valid_features on top_features_0 defined above) in the form of
# a list of column names in the dataframe. top_features are in ascending order of p-values as determined by an F-test
# (f_regression) with a ceiling of p = 0.05. Each loop increments the final index of X_columns, so custom input for
# X_columns should be sorted by the desired order for them to be added. Returns a sorted dictionary of the form
# 'num_features': (MSE, R^2, GridSearchCV().best_params_). Generally has a long runtime.
def hyper_feature(y, X_columns, df, param_dictn, scaling='none', sv=False, splt=0.2,
                  sd=val_seed, ste=val_state, vb=3):
    hyper_rf_dict = {}
    try:
        for i in range(1, len(X_columns) + 1):
            X_trial = df[X_columns[:i]]
            print(f"\nBeginning loop {i} of {len(X_columns)} for columns: {list(X_columns[:i])}:\n")
            opt_params, rsquare, mse = hyper_param(y, X_trial, param_dictn, scl=scaling, save=sv, split_frc=splt,
                                                   seed=sd, state=ste, verbosity=vb)
            hyper_rf_dict[i] = (mse, rsquare, opt_params)
    except KeyboardInterrupt:
        print("\nProcess halted prematurely. Displaying current output.\n")
    finally:
        sorted_hyper_rf_dict = dict(sorted(hyper_rf_dict.items(), key=lambda z: z[1][0]))
        print(sorted_hyper_rf_dict)
        return sorted_hyper_rf_dict


def rf_model_pipeline(raw_features, target, df, param_dictn, k='all', alpha=0.05, sv='', verbosity=3):
    """
    Compiles previous functions into a single pipeline that performs hyperparametric tuning to find the optimal
    random forest model. Prints information and returns a final model. Generally has a long runtime.

        Parameters:
            :param raw_features: If an empty array, will proceed to run on all valid numeric columns in df.
                                 If nonempty, acts as an initial set of candidate features to be verified by
                                 valid_features and further reduced.
            :type raw_features: list
            :param target: Target variable for random forest model.
            :type target: str
            :param df: Dataframe containing at a minimum the variables in raw_features and target.
            :type df: pd.DataFrame
            :param param_dictn: Dictionary of parameter values to be used in GridSearchCV within hyper_param.
            :type param_dictn: dict
            :param k: Number of features selected by SelectKBest in feature_selection. Default 'all'.
            :type k: int or str ('all')
            :param alpha: Cutoff for p-values in feature selection when calculating f_regression and VIF values.
            :type alpha: float
            :param sv: File name to save a csv of the target variable data and X_0 dataframe created by
                       create_y_X_frames. If top_features is reduced from raw_features, create_y_X_frames will be run
                       again and overwrite the save location with the reduced dataframe X_1 and target variable data.
            :type sv: str
            :param verbosity: Controls message output from GridSearchCV (in hyper_param which is used by hyper_feature).
            :type verbosity: int

        Returns:
            :return: rf_model: Optimal random forest model given raw_features, target, df, and param_dictn.
            :rtype: RandomForestRegressor
    """

    display_plots = False
    save_plots = False
    crs_val = True

    # Default values used for a comparison random forest model
    comp_n_est = 400
    comp_m_feat = 8
    comp_m_dep = 8
    comp_m_spl = 8
    comp_m_leaf = 8
    spl = 0.2
    sd = val_seed
    ste = val_state

    if raw_features:
        predictors_initial = valid_features(raw_features, target)
    else:
        predictors_initial = valid_features(list(df.select_dtypes(include=np.number).columns), target)
    y_0, X_0, X_1 = create_y_X_frames(predictors_initial, target, df, True, save_name=sv)
    X_rf = X_1 if raw_features else X_0

    df_features, opt_indp_var = feature_selection(y_0, X_rf, df, k, alpha)
    print(f"\nOptimal independent variables selected by feature_selection:\n{list(opt_indp_var)}\n")

    # Searching for best subset of opt_indp_var independent variables to minimize MSE
    # Extremely long runtime depending on settings
    dictn_hyper_feature = hyper_feature(y_0, opt_indp_var, df, param_dictn, vb=verbosity)
    print(f"\nThe above is the output of hyper_feature containing {{number of features: (MSE, R^2, "
          f"{{RF parameter: optimal value from param_dictn}})}}\n")

    opt_num_features, opt_stats = next(iter(dictn_hyper_feature.items()))
    opt_mse, opt_r2, opt_params = opt_stats
    top_features = opt_indp_var[:opt_num_features]
    print(f"Optimal feature set top_features =\n{list(top_features)}\n")
    print(f"Removes {[x for x in opt_indp_var if x not in top_features]} from opt_indp_var =\n{list(opt_indp_var)}\n")

    if set(top_features) != set(predictors_initial):
        print(f"predictors_initial = \n{predictors_initial}\ndisagrees with the elements found in top_features. "
              f"Overwriting X_1 dataframe with (valid) top_features columns.\n")
        y_0, X_0, X_1 = create_y_X_frames(top_features, target, df, True, save_name=sv)

    print(f"\nComparison random forest model using n_estimators={comp_n_est}, max_features={comp_m_feat}, "
          f"max_depth={comp_m_dep if comp_m_dep else 'None'}, min_samples_split={comp_m_spl}, "
          f"min_samples_leaf={comp_m_leaf}:")
    predictor_rf(y_0, X_1, n_est=comp_n_est, m_feat=comp_m_feat, m_dep=comp_m_dep, m_spl=comp_m_spl, m_leaf=comp_m_leaf,
                 split_frc=spl, seed=sd, state=ste, save=False, output=True,
                 save_plots=False, display_plots=False, cross_val=True)

    # Pass the optimal parameters opt_params to a function that creates a random forest regression model
    c_e = opt_params['n_estimators'] if 'n_estimators' in opt_params else 300
    c_f = opt_params['max_features'] if 'max_features' in opt_params else 9
    c_d = opt_params['max_depth'] if 'max_depth' in opt_params else None
    c_s = opt_params['min_samples_split'] if 'min_samples_split' in opt_params else 5
    c_l = opt_params['min_samples_leaf'] if 'min_samples_leaf' in opt_params else 5

    print(f"\nFinal random forest using n_estimators={c_e}, max_features={c_f}, max_depth={c_d if c_d else 'None'}, "
          f"min_samples_split={c_s}, min_samples_leaf={c_l}:")
    rf_model, rf_r2, rf_mse, rf_mae, rf_ratio, rf_ftr = predictor_rf(y_0, X_1, n_est=c_e, m_feat=c_f, m_dep=c_d,
                                                                     m_spl=c_s, m_leaf=c_l, split_frc=spl, seed=sd,
                                                                     state=ste, save=False, output=True,
                                                                     save_plots=save_plots, display_plots=display_plots,
                                                                     cross_val=crs_val)
    print("Pipeline complete.\n")
    return rf_model


# Crude RFE for a random forest using fixed 'rf_params' on target 'target', storing the results in a dictionary in place
# of sensitivity and patience strategies that were uncooperative. 'features' is intended to start with an initial large
# set, although it can be done with smaller ones too. Generally has a long runtime.
def rf_rfe(features, target, df, rfe_dict, last_score=np.inf, output=True):
    rf_params = {'n_estimators': 500,
                 'max_features': 8,
                 'max_depth': None,
                 'min_samples_split': 5,
                 'min_samples_leaf': 5}
    sorting_score = 'MSE'

    try:
        y, X_0, X_1 = create_y_X_frames(features, target, df, True, save_name='')
        model, r2, mse, mae, ratio, ftr = predictor_rf(y, X_1, rf_params['n_estimators'], rf_params['max_features'],
                                                       rf_params['max_depth'], rf_params['min_samples_split'],
                                                       rf_params['min_samples_leaf'], 0.2, val_seed, val_state,
                                                       False, output, False, False, False)
        rfe_dict[len(X_1.columns)] = {'MSE': mse, 'RMSE_ratio': ratio, 'MAE': mae, 'r2': r2, 'features': ftr}

        current_score = rfe_dict[len(X_1.columns)][sorting_score]
        next_features = ftr[:-1]
        if len(next_features) > 0:
            print(f"\nThis iteration's {sorting_score}: {current_score}.\nLast iteration's "
                  f"{sorting_score}: {last_score}.\nProceeding with updated features: {next_features}\n")
            rf_rfe(next_features, target, df, rfe_dict, current_score, output)
    except KeyboardInterrupt:
        print("Process halted prematurely. Displaying current output.")
    finally:
        sorted_rfe_dict = dict(sorted(rfe_dict.items(), key=lambda z: z[1][sorting_score]))
        print(f"\nRFE dictionary with keys given by the number of features, "
              f"sorted by {sorting_score} in ascending order:\n{sorted_rfe_dict}\n")
        rfe_num_features, rfe_stats = next(iter(sorted_rfe_dict.items()))
        return sorted_rfe_dict, rfe_stats['features']


# Convenience function to run create_y_X_frames for all target variables
def batch_create_frames(cols_a_pop, cols_a_monthly, cols_a_log_m, cols_t_pop, cols_t_art_m, cols_t_art_log_m,
                        df_art=df_a, df_track=df_t):
    y_a_pop0, X_z, X_a_pop0 = create_y_X_frames(cols_a_pop, target_a_pop, df_art, True, '')
    y_a_monthly0, X_z, X_a_monthly0 = create_y_X_frames(cols_a_monthly, target_a_monthly, df_art, True, '')
    y_a_log_m0, X_z, X_a_log_m0 = create_y_X_frames(cols_a_log_m, target_a_log_m, df_art, True, '')

    y_t_pop0, X_z, X_t_pop0 = create_y_X_frames(cols_t_pop, target_t_pop, df_track, True, '')
    y_t_art_m0, X_z, X_t_art_m0 = create_y_X_frames(cols_t_art_m, target_t_art_m, df_track, True, '')
    y_t_art_log_m0, X_z, X_t_art_log_m0 = create_y_X_frames(cols_t_art_log_m, target_t_art_log_m, df_track, True, '')

    return (y_a_pop0, X_a_pop0, y_a_monthly0, X_a_monthly0, y_a_log_m0, X_a_log_m0, y_t_pop0, X_t_pop0, y_t_art_m0,
            X_t_art_m0, y_t_art_log_m0, X_t_art_log_m0)


# Convenience function to run rf_model_pipeline for all target variables
def batch_pipelines(a_cols, t_cols, params, df_art=df_a, df_track=df_t, k='all'):
    rf_model_pipeline(a_cols, target_a_pop, df_art, params, k, verbosity=1)
    rf_model_pipeline(a_cols, target_a_monthly, df_art, params, k, verbosity=1)
    rf_model_pipeline(a_cols, target_a_log_m, df_art, params, k, verbosity=1)

    rf_model_pipeline(t_cols, target_t_pop, df_track, params, k, verbosity=1)
    rf_model_pipeline(t_cols, target_t_art_m, df_track, params, k, verbosity=1)
    rf_model_pipeline(t_cols, target_t_art_log_m, df_track, params, k, verbosity=1)
    return


# Convenience function to run rf_rfe for all target variables
def batch_rfe(a_rfe_cols, t_rfe_cols, df_art=df_a, df_track=df_t, verbose=False):
    rfe_dictn_a_pop, rfe_dictn_a_monthly, rfe_dictn_a_log_m = {}, {}, {}
    rfe_dictn_t_pop, rfe_dictn_t_art_m, rfe_dictn_t_art_log_m = {}, {}, {}

    a_pop_dict, a_pop_ftr = rf_rfe(a_rfe_cols, target_a_pop, df_art, rfe_dictn_a_pop, output=verbose)
    a_monthly_dict, a_monthly_ftr = rf_rfe(a_rfe_cols, target_a_monthly, df_art, rfe_dictn_a_monthly,
                                           output=verbose)
    a_log_m_dict, a_log_m_ftr = rf_rfe(a_rfe_cols, target_a_log_m, df_art, rfe_dictn_a_log_m, output=verbose)

    t_pop_dict, t_pop_ftr = rf_rfe(t_rfe_cols, target_t_pop, df_track, rfe_dictn_t_pop, output=verbose)
    t_art_m_dict, t_art_m_ftr = rf_rfe(t_rfe_cols, target_t_art_m, df_track, rfe_dictn_t_art_m, output=verbose)
    t_art_log_m_dict, t_art_log_m_ftr = rf_rfe(t_rfe_cols, target_t_art_log_m, df_track, rfe_dictn_t_art_log_m,
                                               output=verbose)
    return a_pop_ftr, a_monthly_ftr, a_log_m_ftr, t_pop_ftr, t_art_m_ftr, t_art_log_m_ftr


# Convenience function to run hyper_param for all target variables given respective columns from output of rf_rfe
def batch_hyper_param(cols_a_pop, cols_a_monthly, cols_a_log_m, cols_t_pop, cols_t_art_m, cols_t_art_log_m, params):
    (y_a_pop, X_a_pop, y_a_monthly, X_a_monthly, y_a_log_m, X_a_log_m, y_t_pop, X_t_pop, y_t_art_m, X_t_art_m,
     y_t_art_log_m, X_t_art_log_m) = batch_create_frames(cols_a_pop, cols_a_monthly, cols_a_log_m,
                                                         cols_t_pop, cols_t_art_m, cols_t_art_log_m)

    hyper_param(y_a_pop, X_a_pop, params)
    hyper_param(y_a_monthly, X_a_monthly, params)
    hyper_param(y_a_log_m, X_a_log_m, params)

    hyper_param(y_t_pop, X_t_pop, params)
    hyper_param(y_t_art_m, X_t_art_m, params)
    hyper_param(y_t_art_log_m, X_t_art_log_m, params)
    return


# Convenience function to run predictor_rf for all target variables with model parameters entered manually
# from hyper_param and predictor columns from output of rf_rfe
def batch_predictor_rf(cols_a_pop, cols_a_monthly, cols_a_log_m, cols_t_pop, cols_t_art_m, cols_t_art_log_m, save=False,
                       output=True, save_plots=False, show_plot=False, cross_val=True):
    (y_a_pop, X_a_pop, y_a_monthly, X_a_monthly, y_a_log_m, X_a_log_m, y_t_pop, X_t_pop, y_t_art_m, X_t_art_m,
     y_t_art_log_m, X_t_art_log_m) = batch_create_frames(cols_a_pop, cols_a_monthly, cols_a_log_m,
                                                         cols_t_pop, cols_t_art_m, cols_t_art_log_m)

    predictor_rf(y_a_pop, X_a_pop, 500, 6, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)
    predictor_rf(y_a_monthly, X_a_monthly, 350, 6, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)
    predictor_rf(y_a_log_m, X_a_log_m, 500, 6, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)

    predictor_rf(y_t_pop, X_t_pop, 500, 10, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)
    predictor_rf(y_t_art_m, X_t_art_m, 500, 10, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)
    predictor_rf(y_t_art_log_m, X_t_art_log_m, 350, 6, None, 4, 4, 0.2, val_seed, val_state, save, output, save_plots,
                 show_plot, cross_val)
    return







# Optimal feature sets for each of the target variables as determined by rf_rfe
a_pop_cols = ['log_monthly_listeners_val', 'log_monthly_listeners_mean', 'monthly_listeners_mean',
              'monthly_listeners_val', 'log_monthly_listeners_min', 'monthly_listeners_min',
              'log_monthly_listeners_max', 'monthly_listeners_max', 'followers_max', 'followers_mean', 'followers_val',
              'followers_min', 'monthly_listeners_std_dev', 'featured_count', 'followers_std_dev', 'num_releases',
              'followers_range', 'monthly_listeners_range', 'track_tempo_range', 'track_energy_range',
              'track_instrumentalness_min', 'track_liveness_range', 'last_release', 'track_release_date_max',
              'track_energy_mean', 'track_speechiness_max', 'first_release', 'track_speechiness_mean',
              'track_speechiness_val', 'track_speechiness_min', 'track_release_date_mean', 'track_loudness_max',
              'track_instrumentalness_val', 'track_duration_ms_min', 'track_loudness_min', 'track_duration_ms_max',
              'track_danceability_max', 'track_valence_min', 'track_liveness_max', 'track_tempo_min', 'track_tempo_max',
              'track_valence_val', 'track_acousticness_mean', 'track_danceability_mean', 'track_acousticness_min',
              'track_danceability_min', 'track_acousticness_val', 'track_danceability_val', 'track_release_date_val',
              'track_loudness_std_dev']

a_monthly_cols = ['popularity_val', 'popularity_mean', 'popularity_max', 'popularity_min', 'followers_max',
                  'followers_mean', 'track_popularity_max', 'followers_val', 'followers_min', 'featured_count',
                  'followers_range', 'track_loudness_range', 'track_acousticness_min', 'track_speechiness_min',
                  'track_release_date_mean', 'followers_std_dev', 'track_release_date_min']

a_log_m_cols = ['popularity_max', 'popularity_val', 'popularity_mean', 'popularity_min', 'track_popularity_max',
                'track_popularity_mean', 'followers_max', 'track_popularity_val', 'followers_val',
                'followers_mean', 'followers_min', 'track_popularity_min', 'featured_count', 'followers_std_dev',
                'num_releases', 'followers_range', 'last_release', 'first_release', 'track_speechiness_val',
                'track_speechiness_min', 'track_energy_min', 'track_speechiness_mean', 'track_acousticness_val',
                'track_acousticness_max', 'track_acousticness_mean', 'track_loudness_min',
                'track_danceability_max', 'track_valence_min', 'track_danceability_mean',
                'track_instrumentalness_max', 'track_tempo_max', 'track_danceability_min',
                'track_duration_ms_val', 'track_duration_ms_min']

t_pop_cols = ['count', 'featured', 'artist_featured_count', 'artist_log_monthly_listeners_max',
              'artist_monthly_listeners_max', 'markets', 'artist_monthly_listeners_mean',
              'artist_monthly_listeners_val', 'artist_log_monthly_listeners_mean', 'artist_log_monthly_listeners_val',
              'artist_first_release', 'release_year', 'loudness', 'artist_monthly_listeners_min',
              'artist_followers_min', 'acousticness', 'energy', 'artist_num_releases', 'duration_ms',
              'artist_followers_std_dev']

t_art_m_cols = ['artist_popularity_mean', 'artist_popularity_val', 'artist_popularity_max', 'artist_popularity_min',
                'artist_followers_max', 'artist_followers_val', 'artist_followers_range', 'artist_featured_count',
                'artist_followers_mean', 'artist_followers_std_dev', 'artist_first_release', 'artist_followers_min',
                'artist_pop', 'artist_popularity_std_dev', 'artist_num_tracks']

t_art_log_m_cols = ['artist_popularity_max', 'artist_popularity_val', 'artist_popularity_mean',
                    'artist_popularity_min', 'artist_followers_max', 'artist_followers_val', 'artist_followers_mean',
                    'artist_followers_min', 'popularity', 'artist_featured_count', 'count', 'artist_followers_range',
                    'artist_followers_std_dev', 'artist_num_releases', 'artist_popularity_std_dev',
                    'artist_last_release', 'artist_popularity_range', 'speechiness', 'artist_first_release',
                    'acousticness', 'instrumentalness', 'danceability', 'loudness', 'energy', 'duration_ms', 'tempo',
                    'valence', 'liveness', 'release_year', 'artist_num_tracks', 'musicalkey', 'artist_pop']


# Running functions
'''
info_print(df_a)
info_print(df_t)
extensive_exploration(reduced_a_columns, reduced_t_columns, show_plots=False)
# batch_pipelines(reduced_a_columns, reduced_t_columns, param_dict_small)
# batch_rfe(rfe_cols_a, rfe_cols_t)
# batch_hyper_param(a_pop_cols, a_monthly_cols, a_log_m_cols, t_pop_cols, t_art_m_cols, t_art_log_m_cols, param_dict)
batch_predictor_rf(a_pop_cols, a_monthly_cols, a_log_m_cols, t_pop_cols, t_art_m_cols, t_art_log_m_cols, show_plot=True)
'''








# eof
