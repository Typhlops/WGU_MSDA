import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime


# Spotify featured artist and track data from Kaggle (Sarah Jeffreson, Spotify).
# Located at https://www.kaggle.com/datasets/sarahjeffreson/featured-spotify-artiststracks-with-metadata
# Accessed May 15, 2024.
df_fca = pd.read_csv('data/D214-spotify-files/featured_CLEANED_Spotify_artist_info.csv')
df_ft = pd.read_csv('data/D214-spotify-files/featured_Spotify_track_info.csv')
df_fa = pd.read_csv('data/D214-spotify-files/featured_Spotify_artist_info.csv')

# Spotify random artist and track data from Kaggle (Sarah Jeffreson, Spotify).
# Located at https://www.kaggle.com/datasets/sarahjeffreson/large-random-spotify-artist-sample-with-metadata
# Accessed May 30, 2024.
df_rca = pd.read_csv('data/D214-spotify-files/random_CLEANED_Spotify_artist_info.csv')
df_rt = pd.read_csv('data/D214-spotify-files/random_CLEANED_Spotify_artist_info_tracks.csv')
df_ra = pd.read_csv('data/D214-spotify-files/random_Spotify_artist_info.csv')

# There was one instance of an artist having more than one unique name. Artist id 7mgY992t7YTx6UELsoIMRa was referred
# to by both 'woo!ah!' and 'wooah' in df_fa. The original data files available at the above Kaggle links were modified
# to consistently use only 'woo!ah!'.

pd.set_option('display.max_columns', None)


# fca = featured clean artist, ft = featured track, fa = featured artist, similarly r = random for the others
fca_columns = ['dates', 'ids', 'names', 'monthly_listeners', 'popularity', 'followers', 'genres', 'first_release',
               'last_release', 'num_releases', 'num_tracks', 'playlists_found', 'feat_track_ids']
ft_columns = ['ids', 'names', 'popularity', 'markets', 'artists', 'release_date', 'duration_ms', 'acousticness',
              'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
              'musicalkey', 'musicalmode', 'time_signature', 'count', 'dates', 'playlists_found']
fa_columns = ['dates', 'ids', 'names', 'monthly_listeners', 'popularity', 'followers', 'genres', 'first_release',
              'last_release', 'num_releases', 'num_tracks', 'playlists_found', 'feat_track_ids']

rca_columns = ['ids', 'names', 'popularity', 'followers', 'genres', 'first_release', 'last_release', 'num_releases',
               'num_tracks', 'monthly_listeners']
rt_columns = ['ids', 'names', 'popularity', 'markets', 'artists', 'release_date', 'duration_ms', 'acousticness',
              'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
              'musicalkey', 'musicalmode', 'time_signature']
ra_columns = ['ids', 'names', 'popularity', 'followers', 'genres', 'first_release', 'last_release', 'num_releases',
              'num_tracks', 'monthly_listeners']


# Genre keyword categories and substrings to search for in an artist's listed genres
genre_keywords = {
    'ambient': ['ambient', 'calm', 'relax', 'meditation', 'natur', 'background', 'lofi', 'lo-fi', 'lo fi', 'binaural',
                'easy listening', 'drone', 'chillstep', 'fantasy', 'sleep', 'atmospher', 'healing', 'spa treatment',
                'rain', 'white noise'],
    'asian': ['j-', 'japan', 'anime', 'korea', 'k-rock', 'k-indie', 'k-rap', 'k-pop', 'krown', 'indonesia', 'thai',
              'khmer', 'bali', 'indian ', 'bollywood', 'chinese', 'mandopop', 'cantopop', 'hmong', 'arab', ' k-pop',
              'opm', 'seiyu', 'ryukoka', 'shakuhachi', 'singapore', 'malay'],
    'christian': ['christian', 'gospel', 'worship', 'praise', 'hymn', 'bible'],
    'classical_instrumental': ['classical', 'opera', 'baroque', 'choral', 'choir', 'orchestra', 'instrumental',
                               'renaissance', 'medieval', 'romanticism', 'romantic era', 'impressionism', 'mandolin',
                               'neoclassic', 'neo-classic', 'neo classic', 'post-romantic', 'violin', 'cello', 'piano',
                               'string ', 'symphon', 'chant', 'harpsichord', 'wind ', 'bagpipe', 'military band',
                               'quartet', 'quintet', 'chamber music', 'didgeridoo', 'guitar', 'brass ensemble',
                               'ukulele'],
    'contemporary': ['contemporary', 'modern', 'singer-songwriter', 'world'],
    'country': ['country', 'bluegrass'],
    'electronic': ['electronic', 'electro', 'house', 'techno', 'edm', 'dubstep', 'brostep', 'substep', 'deathstep',
                   'trance', 'synth', 'futur', 'cyberpunk', 'grime', 'hardstyle', 'club', 'ebm', 'zouk', 'dancehall',
                   'downtempo', 'spacewave', 'psytech', 'aussietronica', 'drift phonk', 'moombahton', 'makina', 'acid',
                   'glitchcore', 'psydub', 'psybass', 'gym phonk', 'extratone', 'dembrow', 'chillwave', 'glitch',
                   'hardwave', 'abstract beats', 'breakcore', 'vaporwave', 'psytrance', 'hard dance', 'big room',
                   'electronica'],
    'euro': ['uk ', 'brit', 'swiss', 'french', 'franc', 'german', 'deutsch', 'italian', 'italo', 'irish', 'gaelic',
             'welsh', 'celt', 'swedish', 'sweden', 'norwegian', 'nordic', 'danish', 'scandinavian', 'munich', 'austria',
             'hungarian', 'polish', 'poland', 'ukrain', 'russia', 'euro', 'scottish', 'greek', 'belgian', 'dutch',
             'iceland', 'turkish', 'saxon', 'finnish', 'lithuania', 'berlin', 'oslo'],
    'folk': ['folk', 'americana', 'traditional', 'bluegrass', 'banjo', 'celtic', 'contra dance', 'fiddle'],
    'hip-hop': ['hip-hop', 'hiphop', ' hop', 'hip ', 'trap', 'rap', 'boom bap', 'drill', 'abstractro', 'ghettotech',
                'hiplife', 'phonk'],
    'indie_alternative': ['indie', 'alternative', 'alt '],
    'jazz': ['jazz', 'blues', 'swing', 'bebop', 'ragtime', 'saxophone', 'american primitive', 'hard bop', 'dixieland'],
    'latin': ['latin', 'mexic', 'brazil', 'brasil', 'espanol', 'sierreno', 'argentin', 'chile', 'carimbo', 'salsa',
              'flamenco'],
    'metal': ['metal', 'grindcore', 'screamo', 'thrash', 'death'],
    'pop': ['pop', 'new wave', 'new-wave', 'easycore', 'idol', 'seiyu'],
    'reggae_soul': ['reggae', 'soul', 'funk', 'ska', 'r&b', 'disco', 'motown', 'kampa', 'haiti', 'jamaica', 'samba',
                    'steel drum', 'carib'],
    'rock': ['rock', 'progressive', 'goth', 'punk', 'grunge', 'hardcore', 'emo', 'psych']
}


# Creating a data source column 'featured' (1 or 0) and a 'tracks' column in the artist dataframes to be filled in
df_fa['tracks'] = [[] for _ in range(len(df_fa))]
df_fa.insert(0, 'featured', 1)
df_ft.insert(0, 'featured', 1)

df_ra['tracks'] = [[] for _ in range(len(df_ra))]
df_ra.insert(0, 'featured', 0)
df_rt.insert(0, 'featured', 0)


# Creating sets for easier identification in later functions creating dictionaries of form {artist_id: tracks}
featured_artists = set(df_fa['ids'].unique())  # len 10617
featured_clean_artists = set(df_fca['ids'].unique())  # len 7603
random_artists = set(df_ra['ids'].unique())  # len 37000
random_clean_artists = set(df_rca['ids'].unique())  # len 15027
track_set_r = set(df_rt['ids'].unique())  # len 15013
track_set_f = set(df_ft['ids'].unique())  # len 15052

# Sets of artists (tracks) appearing in both df_fa and df_ra (df_ft and df_rt)
dup_artists = featured_artists.intersection(random_artists)  # len 450
dup_tracks = track_set_f.intersection(track_set_r)  # len 23

# Identifying tracks with null values
null_tracks_r = df_rt[df_rt.isna().any(axis=1)]
null_tracks_f = df_ft[df_ft.isna().any(axis=1)]
rt_null_set = set(null_tracks_r['ids'])
ft_null_set = set(null_tracks_f['ids'])
print("Set of null tracks in df_ft that exist in df_rt tracks: ", ft_null_set.intersection(track_set_r))
print("Set of null tracks in df_rt that exist in df_ft tracks: ", rt_null_set.intersection(track_set_f))
print(f"\nArtists from tracks containing null values in df_ft:\n{null_tracks_f['artists']}\n")
print("Associated information for those artists in df_fa:\n")
print(df_fa[df_fa['ids'].isin(null_tracks_f['artists'])][['dates', 'ids', 'names', 'monthly_listeners', 'popularity',
                                                          'followers', 'first_release', 'playlists_found']])

# Looking for duplicates in df_fa
print(f"\nDuplicates on df_fa[['dates', 'ids']]: {df_fa.duplicated(subset=['dates', 'ids'], keep=False).sum()}")
print(f"Number of unique values in df_fa['ids']: {len(df_fa['ids'].unique())}")
print(f"Number of unique values in df_fa[['ids', 'names']]: {len(df_fa[['ids', 'names']].drop_duplicates())}")
print(f"Do 'ids' in df_fa have only one name?: "
      f"{set(df_fa['ids'].unique()) == set(df_fa[['ids', 'names']].drop_duplicates()['ids'])}\n")

# Looking for duplicates in df_ra
print(f"Duplicates on df_ra['ids']: {df_ra.duplicated(subset=['ids'], keep=False).sum()}")
print(f"Number of unique values in df_ra['ids']: {len(df_ra['ids'].unique())}")
print(f"Number of unique values in df_ra[['ids', 'names']]: {len(df_ra[['ids', 'names']].drop_duplicates())}")
print(f"Do 'ids' in df_ra have only one name?: "
      f"{set(df_ra['ids'].unique()) == set(df_ra[['ids', 'names']].drop_duplicates()['ids'])}\n")


# Dictionary of dataframes by string identifier
dictn_df = {'fca': df_fca, 'ft': df_ft, 'fa': df_fa, 'rca': df_rca, 'rt': df_rt, 'ra': df_ra}

# Initializing dictionaries (and a set of artists listed in tracks.artists that aren't found in the artist
# dataframes df_fa or df_ra). Described in further detail in respective functions that fill them.
artists_w_no_data_set = set()
artists_w_no_data, art_track_dict_f, art_track_dict_r = {}, {}, {}

art_feat_track_dict, art_feat_track_w_dates_dict, null_track_dict_f = {}, {}, {}

art_genre_dict_f, art_genre_dict_r = {}, {}


# Looks through each dataframe located in dict_df to look for null values and their respective ids, presenting them
# as a set. Also looks for rows duplicating the same id then prints results of .describe().
def find_nulls_dupes(dict_df):
    exceptions = ['genres']  # Too many null values in 'genres' for it to be helpful to print their ids
    dict_null = {}
    for fr in dict_df:
        print(f"\n-----------------------------\nNulls and duplicates for df_{fr}:")
        null_ids = set()
        relevant_exceptions = set(exceptions).intersection(dict_df[fr].columns)
        for coln in dict_df[fr].columns:
            # print(dict_df[fr][coln].value_counts())
            null_count = dict_df[fr][coln].isna().sum()
            if null_count > 0:
                print(f"Nulls in column {coln} for df_{fr}: {null_count}")
                if coln not in relevant_exceptions:
                    null_rows = dict_df[fr][dict_df[fr][coln].isna()]['ids']
                    null_ids.update(set(null_rows))
        dict_null[fr] = null_ids
        tmp_len = len(relevant_exceptions)
        print(f"\nThe {len(null_ids)} null id values {'(other than ' if tmp_len > 0 else ''}"
              f"{relevant_exceptions if tmp_len > 0 else ''}{') ' if tmp_len > 0 else ''}for df_{fr}: \n{null_ids}")
        print(f"Duplicates for df_{fr}: {dict_df[fr].duplicated(['ids']).sum()}\n----------------------")
        print(dict_df[fr].describe(), "\n-----------------------------")
    return dict_null


# Fills dictionaries art_track_dict_f (featured/ft) and art_track_dict_r (random/rt) with track ids associated to any
# artist present in track.artists (of the form {'artist.ids': tracks.ids}). If the artist is not present in
# featured_artists nor random_artists (i.e. df_fa and df_ra ids respectively), they are instead added to
# artists_w_no_data with that track_id, Null tracks are excluded and identified by being present in either
# ft_null_set or rt_null_set. Note that as there are only 23 tracks shared between the featured and random track
# datasets, none of which had null values that had eligible data in its twin, a simplified approach is used ensuring
# that the id is found in neither set. Should that change, use the function dup_validation further below to identify
# these discrepancies and correct them.
def assign_artists_to_tracks(track_id, track_artists):
    if track_id not in ft_null_set and track_id not in rt_null_set:
        artists_list = track_artists.split(", ") if ", " in track_artists else [track_artists]
        for artist in artists_list:
            if artist in featured_artists:
                art_track_dict_f.setdefault(artist, []).append(track_id)
            if artist in random_artists:
                art_track_dict_r.setdefault(artist, []).append(track_id)
            if (artist not in featured_artists) and (artist not in random_artists):
                artists_w_no_data.setdefault(artist, []).append(track_id)
                artists_w_no_data_set.add(artist)
    return


# Fills art_feat_track_dict ({(artist.ids, artist.dates): featured track ids on that date with no null values})
# and art_feat_track_w_dates_dict ({(artist.ids, artist.dates): featured track ids on that date with no null values}).
# Artists associated with featured tracks containing nulls are recorded in null_track_dict_f.
def make_art_f_track_dicts_rmv_dupes(date, art_id, f_track_ids):
    clean_f_track_ids = []
    art_feat_track_dict.setdefault(art_id, [])
    existing_tracks = art_feat_track_dict[art_id]
    tracks_w_nulls = null_track_dict_f[art_id] if art_id in null_track_dict_f else []

    f_track_str = f_track_ids.split(", ") if ", " in f_track_ids else [f_track_ids]
    for s in f_track_str:
        if s not in ft_null_set:
            if s not in clean_f_track_ids:
                clean_f_track_ids.append(s)
            if s not in existing_tracks:
                existing_tracks.append(s)
        else:
            if s not in tracks_w_nulls:
                tracks_w_nulls.append(s)

    if len(tracks_w_nulls) > 0:
        null_track_dict_f.setdefault(art_id, [])
        null_track_dict_f[art_id] = tracks_w_nulls
    art_feat_track_w_dates_dict[(art_id, date)] = clean_f_track_ids
    art_feat_track_dict[art_id] = existing_tracks
    if len(clean_f_track_ids) == 0:
        return None
    else:
        return ", ".join(clean_f_track_ids)


# Extracts genres separated by commas in artist dataframes into the dictionaries art_genre_dict_f (featured)
# and art_genre_dict_r (random), using the form {artist.ids: set(genres)}.
def genre_extract(art_id, genres, dictn):
    dictn.setdefault(art_id, set())
    genre_arr = genres.split(", ") if ", " in genres else [genres]
    for genre in genre_arr:
        dictn[art_id].add(genre)
    return


# There were 19 instances where a featured artist in df_fa had featured tracks that when checked for consistency in
# the tracks dataframe (df_ft) were not present in tracks.artists. When update=True, the track data in
# df_tr (defaults to df_ft) is modified to include the uncredited artist. Additionally, removes duplicated
# track ids in row.feat_track_ids.
def cross_val_feat_art(group, df_tr=df_ft, update=True):
    id_col = 'ids'
    art_col = 'artists'
    track_id_col = 'feat_track_ids'
    group_id = group[id_col].values[0]
    track_vals = group[track_id_col].values
    artist_feat_tracks = set()

    for record in track_vals:
        if isinstance(record, str):
            tracks_arr = record.split(", ") if ", " in record else [record]
            artist_feat_tracks.update(set(tracks_arr))
    for track in artist_feat_tracks:
        track_artists_raw = df_tr[df_tr[id_col] == track][art_col].values[0]
        track_artists = track_artists_raw.split(", ") if ", " in track_artists_raw else [track_artists_raw]
        if group_id not in track_artists:
            print(f"\nWarning: artist {group_id} not credited in track {track}.")
            if update:
                print("Updating track data.")
                df_tr.loc[df_tr[id_col] == track, art_col] = ", ".join(track_artists + [group_id])
    return


# Looks for discrepancies between duplicate entries. Intended to be used with duplicate track ids.
def dup_validation(id_, df_r=df_rt, df_f=df_ft, id_col='ids'):
    comp_cols_str = df_r.select_dtypes(include='object')
    comp_cols_nmr = df_r.select_dtypes(exclude='object')
    conflict_cols = []
    for coln in comp_cols_str:
        if df_r[df_r[id_col] == id_][coln].values != df_f[df_f[id_col] == id_][coln].values:
            conflict_cols.append(coln)
    for coln in comp_cols_nmr:
        diff = abs(df_r[df_r[id_col] == id_][coln].values[0] - df_f[df_f[id_col] == id_][coln].values[0])
        if diff > 0.001:
            conflict_cols.append((coln, diff))
    print(f"{len(conflict_cols)} discrepancies found for track id {id_} on {conflict_cols}\n")


# Convenient shortcut function without modifying original data files as cross_val_feat_art takes ~16 seconds to complete
def fill_uncredited_fast():
    uncredited_artists = ['4bQ8TomvPzE2t8taFNpG03', '3hteYQFiMFbJY7wS0xDymP', '42UC2FPFpWl0phKNeoDvtH',
                          '6EP96GaItADv1rNqR2oGIR', '2U90T4sTIVVjC3klG9QrHW', '1Z9AMFrgGmPJAqLqVTzrHZ',
                          '7dMgnhd87CgdkOoh9V5p1t', '2o2AdpYle5NrFYEU7AoMOe', '5BcAKTbp20cv7tC5VqPFoC',
                          '2KbHSD7lPrLBEFNl05t6GV', '52K2wI1tA5kaRyQUKVhOaJ', '4lqEqZ0gMse3BvOC1YehXP',
                          '45P7v25izf78nRf3bgfYI8', '0W8RdNurhRPm7GJVoPYiea', '65KzmKTmARM8EBVoxki2gn',
                          '7tvN69rnIiIzepCwS9Kew5', '3HQIkVkhoARQMb0XlvyUKL', '6Ms6d5LeChr270J2jRTEUb',
                          '3K85LdQnIv6Vb2hR4mDeVe']

    uncredited_artists_tracks = ['6DJpnE5UQZi7zIZIVYwnMe', '21f4TjwWXbnqg8d5R6oq8Y', '7eMjdhp767PF9F8cAbIyJr',
                                 '0LOjVJaTDqXyu37cjUYpT2', '5OOSqqoKCL4s7WIFNbSZPD', '7CWuyO1HFwgnZaT7BComle',
                                 '0gSgvkA2ALIdMmYKWgwsfx', '594MueE2n17NeaZYiGdhYP', '3bidbhpOYeV4knp8AIu8Xn',
                                 '63QYGp2ZTqDrEhNWGAnURD', '6XbfAA4NymHlQ52zeMitGK', '1hsFQ9lhiw39h39s3zeHUe',
                                 '3as75756VNBEvwJPSVfvBR', '1RR3PCooe7Uhak8sUP3OQB', '3mUMCfl8kOxAR6rPEAl7bv',
                                 '6ibDVMcMUNqZ5eXT9sD4Vy', '3u25dhBZ80h0TFewO29Wqb', '7gg77ywX9Y0n6NPIIETWus',
                                 '2qU4cGVCZ6CJQ8RjMHvGvp']

    uncredited_artist_track_tuples = list(zip(uncredited_artists, uncredited_artists_tracks))

    for artist, track in uncredited_artist_track_tuples:
        df_ft.loc[df_ft['ids'] == track, 'artists'] = df_ft[df_ft['ids'] == track]['artists'].values[0] + ', ' + artist
    return


# Filters a single string containing all the artist's genres (with or without commas) to identify broad genre
# categories as listed at the top of the file in genre_keywords. A special case handles 'folk-pop' being captured
# when searching for 'k-pop'. The allow_other parameter uses the placeholder other_genre_name='other' when an
# artist's non-empty genre string finds no matches with the substring searches from genre_keywords. If the artist has
# non-null entries in their 'playlists_found' column and no matching genre categories (with the exception of 'other')
# were previously found, the playlists are scanned to attempt to determine potential genres for the artist. Returns
# an array of the artist's genres (e.g. ['rock', 'pop']) or None if no matches are found (with allow_other=False).
def genre_filter(s, playlist='', allow_other=False):
    other_genre_name = 'other'
    genre_categories = []

    if isinstance(s, str) and len(s) > 0:
        sl = s.lower()
        for genre_group in genre_keywords:
            for keyword in genre_keywords[genre_group]:
                if keyword in sl:
                    genre_categories.append(genre_group)
                    break
        if 'folk-pop' in sl:
            asian_remove = True
            asian_reduced = [w for w in genre_keywords['asian'] if w != 'k-pop']
            for word in asian_reduced:
                if word in sl:
                    asian_remove = False
                    break
            if asian_remove:
                genre_categories.remove('asian')
        if allow_other and len(genre_categories) == 0:
            genre_categories.append(other_genre_name)

    if len(playlist) > 0 and (len(genre_categories) == 0 or genre_categories[0] == other_genre_name):
        # could quickly check if playlist includes korean characters but the few relevant playlists all have
        # english translations included with 'k-pop' and similar terms that would be captured
        pl = playlist.lower()
        for genre_group in genre_keywords:
            for keyword in genre_keywords[genre_group]:
                if keyword in pl:
                    genre_categories.append(genre_group)
                    break
        if 'folk-pop' in pl:
            asian_remove = True
            asian_reduced = [w for w in genre_keywords['asian'] if w != 'k-pop']
            for word in asian_reduced:
                if word in pl:
                    asian_remove = False
                    break
            if asian_remove:
                genre_categories.remove('asian')

    # print(genre_categories)
    if len(genre_categories) == 0:
        genre_categories = None
    return genre_categories


# Used to create aggregated dataframe of artist data by artist id (after concatenating both featured and random
# artist dataframes). Returns a pandas series of 'result' including the number of times the artist appeared in
# featured playlists within the datasets, the dates they were featured, their id, name, genre categories (as
# described by genre_keywords) separated by commas within a single string, the minimum (valid) first_release year,
# maximal (valid) values for last_release and num_releases and num_tracks, all playlists they were found in,
# their (valid) featured track ids, the ids of any track they're credited in within the track dataframes,
# and the one-hot encoding of their genre categories. Additionally, for monthly_listeners, popularity, followers,
# and log_monthly_listeners (the log_10 of monthly_listeners), their statistical results are returned in the form of
# a dictionary described by stats_dict. Data for each track within the artist's 'tracks' column was used to create
# summary statistics of track data columns (e.g. valence and tempo).
def artist_agg(group, verbose=False):
    group_by_col = 'ids'
    dates_col = 'dates'
    genres_col = 'genres'
    featured_col = 'featured'
    playlists_col = 'playlists_found'
    monthly_listeners_col = 'monthly_listeners'
    first_col = 'first'
    last_col = 'last'
    columns_str = ['genres', 'playlists_found']
    columns_nmr_agg = ['monthly_listeners', 'popularity', 'followers']
    columns_nmr_union = ['first_release', 'last_release', 'num_releases', 'num_tracks']
    comparison_columns = columns_nmr_agg
    column_min_vals = {'monthly_listeners': 0, 'followers': 0, 'first_release': 0, 'last_release': 0, 'num_releases': 0,
                       'num_tracks': 0}
    stats_dict = {'mean': np.mean, 'median': np.median, 'std_dev': np.std, 'min': np.min, 'max': np.max}
    log_function = np.log10

    track_columns_agg = ['popularity', 'release_date', 'duration_ms', 'acousticness', 'danceability', 'energy',
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    track_columns_mode = ['musicalkey', 'musicalmode', 'time_signature']
    track_min_vals = {'popularity': 0.0, 'release_date': 0.0, 'duration_ms': 0.0, 'acousticness': 0.0,
                      'danceability': 0.0, 'energy': 0.0, 'instrumentalness': 0.0, 'liveness': 0.0, 'loudness': -60.0,
                      'speechiness': 0.0, 'tempo': 0.0, 'valence': 0.0, 'musicalkey': 0.0, 'musicalmode': 0.0,
                      'time_signature': 0.0}
    track_max_vals = {'popularity': 100, 'release_date': datetime.now().year, 'duration_ms': 1e7, 'acousticness': 1.0,
                      'danceability': 1.0, 'energy': 1.0, 'instrumentalness': 1.0, 'liveness': 1.0, 'loudness': 5.0,
                      'speechiness': 1.0, 'tempo': 300.0, 'valence': 1.0, 'musicalkey': 12, 'musicalmode': 1.0,
                      'time_signature': 5.0}
    track_id_col = 'ids'
    track_release_date_col = 'release_date'
    track_columns = track_columns_agg + track_columns_mode
    df_f = df_ft
    df_r = df_rt
    track_data_arrays = {}
    track_data_stats = {}

    group_id = group[group_by_col].values[0]
    datetimes_arr = [datetime.strptime(date, '%m/%d/%Y') for date in group[dates_col] if pd.notnull(date)]
    coln_set_unions_dict = {}
    nmr_agg_dict = {}
    first_last_agg_dict = {}
    all_tracks = set()
    non_redundant = True
    num_decimals = 4

    for index, row in group.iterrows():
        if row[featured_col] == 0 and 1 in group[featured_col].values:
            val_r = row[comparison_columns].values
            dates_f = group[group[featured_col] == 1][dates_col].values
            vals_f = group[group[featured_col] == 1][comparison_columns].values
            for idx, val_f in enumerate(vals_f):
                vals_diff = abs(val_r - val_f)
                thresh = 0.005 * val_f
                if np.all(pd.notnull(vals_diff)):
                    if np.all(vals_diff < thresh):
                        if verbose:
                            print(f"\nMatch found for {group_id} on {dates_f[idx]}\n")
                        non_redundant = False
                        break
        for cl in columns_str:
            data = row[cl]
            coln_set_unions_dict.setdefault(cl, set())
            if isinstance(data, str):
                data_arr = data.split(", ") if ", " in data else [data]
                coln_set_unions_dict[cl].update(set(data_arr))
        if non_redundant:
            for cl in columns_nmr_agg:
                data = row[cl]
                coln_set_unions_dict.setdefault(cl, [])
                if isinstance(data, int) or (isinstance(data, float) and not np.isnan(data)):
                    if cl in column_min_vals:
                        thresh_min = column_min_vals[cl]
                        if data > thresh_min:
                            coln_set_unions_dict[cl].append(data)
                        else:
                            if pd.notnull(row[dates_col]):
                                formatted_datetime = datetime.strptime(row[dates_col], '%m/%d/%Y')
                                if formatted_datetime in datetimes_arr:
                                    datetimes_arr.remove(formatted_datetime)
                                    if verbose:
                                        print(f"\nRemoved date {formatted_datetime.strftime('%m/%d/%Y')} "
                                              f"for considering first and last entries on group id {group_id}\n")
                    else:
                        coln_set_unions_dict[cl].append(data)
                else:
                    if pd.notnull(row[dates_col]):
                        formatted_datetime = datetime.strptime(row[dates_col], '%m/%d/%Y')
                        if formatted_datetime in datetimes_arr:
                            datetimes_arr.remove(formatted_datetime)
                            if verbose:
                                print(f"\nRemoved date {formatted_datetime.strftime('%m/%d/%Y')} "
                                      f"for considering first and last entries on group id {group_id}\n")
        for cl in columns_nmr_union:
            data = row[cl]
            coln_set_unions_dict.setdefault(cl, set())
            thresh_min = column_min_vals[cl]
            if isinstance(data, int) or (isinstance(data, float) and not np.isnan(data)):
                if data > thresh_min:
                    coln_set_unions_dict[cl].add(data)

    if datetimes_arr:
        min_date = min(datetimes_arr).strftime('%m/%d/%Y')
        max_date = max(datetimes_arr).strftime('%m/%d/%Y')
        for index, row in group.iterrows():
            if pd.notnull(row[dates_col]):
                formatted_date = datetime.strptime(row[dates_col], '%m/%d/%Y').strftime('%m/%d/%Y')
                if formatted_date == min_date:
                    for cl in columns_nmr_agg:
                        first_last_agg_dict.setdefault(cl, [-1, -1])
                        first_last_agg_dict[cl][0] = row[cl]
                if formatted_date == max_date:
                    for cl in columns_nmr_agg:
                        first_last_agg_dict.setdefault(cl, [-1, -1])
                        first_last_agg_dict[cl][1] = row[cl]

    if group_id in art_genre_dict_f:
        coln_set_unions_dict[genres_col].update(art_genre_dict_f[group_id])
    if group_id in art_genre_dict_r:
        coln_set_unions_dict[genres_col].update(art_genre_dict_r[group_id])
    filtered_genres = genre_filter(", ".join(coln_set_unions_dict[genres_col]),
                                   ", ".join(coln_set_unions_dict[playlists_col]))

    for dic in [art_track_dict_f, art_track_dict_r, art_feat_track_dict]:
        if group_id in dic:
            all_tracks.update(dic[group_id])

    if 1 in group[featured_col].values:
        # result_featured = 1
        result_dates = ", ".join(date for date in group[dates_col] if pd.notnull(date))
        result_playlists = ", ".join(coln_set_unions_dict[playlists_col])
        result_feat_tracks = ", ".join(art_feat_track_dict[group_id])
    else:
        # result_featured = 0
        result_dates = None
        result_playlists = None
        result_feat_tracks = None

    for cl in columns_nmr_agg:
        tmp_arr = coln_set_unions_dict[cl]
        col_stats = None
        if len(tmp_arr) > 1:
            col_val = tmp_arr
            col_stats = {k: round(fn(tmp_arr), num_decimals) for k, fn in stats_dict.items()}
            if datetimes_arr:
                col_stats[first_col] = first_last_agg_dict[cl][0]
                col_stats[last_col] = first_last_agg_dict[cl][1]
            else:
                col_stats[first_col] = None
                col_stats[last_col] = None
        elif len(tmp_arr) == 1:
            col_val = tmp_arr[0]
        else:
            col_val = None
        nmr_agg_dict[cl] = (col_val, col_stats)

    pre_m_l_log_arr = coln_set_unions_dict[monthly_listeners_col]
    m_l_log_arr = log_function(pre_m_l_log_arr)
    m_l_log_stats = None
    if len(m_l_log_arr) > 1:
        m_l_log_val = np.round(m_l_log_arr, 4)
        m_l_log_stats = {k: round(fn(m_l_log_arr), num_decimals) for k, fn in stats_dict.items()}
        if datetimes_arr:
            m_l_log_stats[first_col] = round(log_function(first_last_agg_dict[monthly_listeners_col][0]), num_decimals)
            m_l_log_stats[last_col] = round(log_function(first_last_agg_dict[monthly_listeners_col][1]), num_decimals)
        else:
            m_l_log_stats[first_col] = None
            m_l_log_stats[last_col] = None
    elif len(m_l_log_arr) == 1:
        m_l_log_val = round(m_l_log_arr[0], num_decimals)
    else:
        m_l_log_val = None

    if all_tracks:
        for cl in track_columns:
            track_data_arrays.setdefault(cl, [])
        for track in all_tracks:
            track_info = df_f.loc[df_f[track_id_col] == track]
            if track_info.empty:
                track_info = df_r.loc[df_r[track_id_col] == track]
            if track_info.empty:
                print(f"\nTrack {track} not found in either track dataframe.\n")
            else:
                for coln in track_columns:
                    valid_data = True
                    track_data = track_info[coln].values[0]
                    if track_data is not None:
                        if coln == track_release_date_col:
                            try:
                                release_year = datetime.strptime(track_data, '%Y-%m-%d').year
                            except ValueError:
                                try:
                                    release_year = datetime.strptime(track_data, '%Y-%m').year
                                except ValueError:
                                    try:
                                        release_year = datetime.strptime(track_data, '%Y').year
                                    except ValueError:
                                        release_year = None
                                        valid_data = False
                                        print(f"\nDate format of release_date {track_data} not parsed.\n")
                            track_data = release_year
                        if isinstance(track_data, (int, float, np.integer, np.floating)):
                            if coln in track_min_vals.keys():
                                if track_data < track_min_vals[coln]:
                                    valid_data = False
                            if coln in track_max_vals.keys():
                                if track_data > track_max_vals[coln]:
                                    valid_data = False
                        if valid_data:
                            track_data_arrays[coln].append(track_data)

    if track_data_arrays:
        for cl in track_columns_agg:
            tmp_arr = track_data_arrays[cl]
            col_stats = None
            if len(tmp_arr) > 1:
                col_val = tmp_arr
                col_stats = {k: round(fn(tmp_arr), num_decimals) for k, fn in stats_dict.items()}
            elif len(tmp_arr) == 1:
                col_val = tmp_arr[0]
            else:
                col_val = None
            track_data_stats[cl] = (col_val, col_stats)
        for cl in track_columns_mode:
            tmp_arr = track_data_arrays[cl]
            col_stats = None
            if len(tmp_arr) > 1:
                col_val = tmp_arr
                col_stats = {'mode': stats.mode(tmp_arr)[0]}
            elif len(tmp_arr) == 1:
                col_val = tmp_arr[0]
            else:
                col_val = None
            track_data_stats[cl] = (col_val, col_stats)

    result = {
        'featured_count': group[featured_col].sum(),
        'dates': result_dates,
        'ids': group_id,
        'names': ", ".join(set(group['names'])),
        'monthly_listeners': nmr_agg_dict[monthly_listeners_col],
        'popularity': nmr_agg_dict['popularity'],
        'followers': nmr_agg_dict['followers'],
        'genres': ", ".join(sorted(filtered_genres)) if filtered_genres else None,
        'first_release': min(coln_set_unions_dict['first_release']) if coln_set_unions_dict['first_release'] else None,
        'last_release': max(coln_set_unions_dict['last_release']) if coln_set_unions_dict['last_release'] else None,
        'num_releases': max(coln_set_unions_dict['num_releases']) if coln_set_unions_dict['num_releases'] else None,
        'num_tracks': max(coln_set_unions_dict['num_tracks']) if coln_set_unions_dict['num_tracks'] else None,
        'playlists_found': result_playlists,
        'feat_track_ids': result_feat_tracks,
        'tracks': ", ".join(all_tracks) if all_tracks else None,
        'log_monthly_listeners': (m_l_log_val, m_l_log_stats),
        'track_stats': track_data_stats if track_data_arrays else None,
    }

    for genre_category in genre_keywords.keys():
        result.setdefault(genre_category, 0)
    if filtered_genres:
        for genre_category in genre_keywords.keys():
            if genre_category in filtered_genres:
                result[genre_category] = 1

    if verbose:
        print(result)
    return pd.Series(result)


# Unwinds dictionaries of statistical values created by artist_agg. Entries where no statistical data is present use
# the suffix 'val'. For a consistent column with no null values, entries that do have statistical data take the value
# from rep_val (defaults to 'median') and store it in the same 'val' key. Returns a pandas series to be used with
# functions further below.
def expand_stat_dict(x, rep_val='median'):
    singular_col_suffix = 'val'
    if x[1] is None:
        data = pd.Series({singular_col_suffix: x[0]})
    else:
        if rep_val not in x[1]:
            rep_val = 'mode'
        x[1][singular_col_suffix] = x[1][rep_val]
        data = pd.Series(x[1])
    return data


# Used to fill null values in the various statistical columns for aggregate group ids that only had 0 or 1 value. The
# standard deviations are set to 0 and the terms found in stat_terms are filled in with what's found in their 'val'
# counterpart for that prefix (found in val_prefixes e.g. tempo and popularity).
def fill_nulls(row, all_df_cols):
    suffix_term = '_val'
    valence_term = 'valence'
    std_dev_term = 'std_dev'
    stat_terms = ['mean', 'median', 'std_dev', 'min', 'max', 'mode', 'first', 'last']
    val_prefixes = {s.removesuffix(suffix_term) for s in all_df_cols if (suffix_term in s and valence_term not in s)
                    or ((valence_term + suffix_term) in s)}

    for prefix in val_prefixes:
        col_to_fill_from = prefix + suffix_term
        to_fill = [prefix + '_' + stat_term for stat_term in stat_terms]
        for col in to_fill:
            if col in row:
                if pd.isnull(row[col]):
                    if std_dev_term in col:
                        row[col] = 0
                    else:
                        row[col] = row[col_to_fill_from]
    return row


# Concatenates the featured and random artist dataframes (df_f=df_fa, df_r=df_ra) and runs the artist_agg function to
# create a new dataframe df_agg, which is then saved to a csv file and returned.
def generate_raw_agg_df(save_csv=False, verbose=False, df_f=df_fa, df_r=df_ra):
    id_col = 'ids'
    raw_agg_save_name = 'raw_all_Spotify_artist_info.csv'

    df_art = pd.concat([df_f, df_r]).reset_index(drop=True)
    df_agg = df_art.groupby(by=id_col, as_index=False)[df_art.columns].apply(lambda group: artist_agg(group, verbose))
    if save_csv:
        print(f"\nSaving raw aggregate dataframe to {raw_agg_save_name}.\n")
        df_agg.to_csv(raw_agg_save_name)
    return df_agg


# Cleans the aggregated artist dataframe returned by generate_raw_agg_df by first removing null values found in columns
# null_columns_to_drop. Columns in cast_int_cols are converted from float64 to int64. For usability in modeling,
# the statistical dictionary columns in coiled_dict_cols are unwound using expand_stat_dict. Newly created null
# values from unwinding the dictionaries are dropped with the columns found in null_columns_to_drop_expand. Null
# statistical values are then filled using fill_nulls. The (cleaned) compact and unwound versions of the aggregate
# dataframe are then saved to csv and the unwound (expanded) one is returned.
def clean_raw_agg_df(agg_df, save_csv=False, rep_stat_val='median'):
    null_columns_to_drop = ['monthly_listeners', 'followers', 'first_release', 'last_release', 'num_releases',
                            'num_tracks', 'tracks', 'genres']
    null_columns_to_drop_expand = ['monthly_listeners_val', 'followers_val', 'first_release', 'last_release',
                                   'num_releases', 'num_tracks', 'tracks', 'genres']
    cast_int_cols = ['first_release', 'last_release', 'num_releases', 'num_tracks']
    coiled_dict_cols = ['monthly_listeners', 'popularity', 'followers', 'log_monthly_listeners', 'track_stats']
    stats_col_suffix = '_stats'
    compact_agg_save_name = 'compact_all_Spotify_artist_info.csv'
    final_agg_save_name = 'all_Spotify_artist_info.csv'

    agg_df = agg_df.dropna(subset=null_columns_to_drop)
    with pd.option_context('mode.copy_on_write', True):
        for col in cast_int_cols:
            try:
                agg_df[col] = agg_df[col].apply(lambda x: int(x))
            except TypeError or ValueError:
                print(f"Unable to convert column {col} from float to int\n")

    df_expand = agg_df
    for cl in coiled_dict_cols:
        if stats_col_suffix in cl:
            df_series = df_expand[cl].apply(pd.Series)
            df_series.columns = cl.replace(stats_col_suffix, '_') + df_series.columns
        else:
            df_series = df_expand[cl].apply(lambda x: expand_stat_dict(x, rep_stat_val))
            df_series.columns = cl + '_' + df_series.columns
        df_expand = df_expand.drop(columns=[cl]).join(df_series)

    track_stat_expanded_cols = [c for c in df_expand.columns if 'track_' in c and '_track' not in c]
    for cl in track_stat_expanded_cols:
        df_series = df_expand[cl].apply(lambda x: expand_stat_dict(x, rep_stat_val))
        df_series.columns = cl + '_' + df_series.columns
        df_expand = df_expand.drop(columns=[cl]).join(df_series)

    df_expand = df_expand.dropna(subset=null_columns_to_drop_expand)
    df_expand = df_expand.apply(lambda row: fill_nulls(row, list(df_expand.columns)), axis=1)

    if save_csv:
        print(f"Saving compact aggregate dataframe with nulls removed to {compact_agg_save_name}.\n")
        agg_df.to_csv(compact_agg_save_name)
        print(f"Saving expanded aggregate dataframe with nulls removed to {final_agg_save_name}.\n")
        df_expand.to_csv(final_agg_save_name)
    return df_expand


# Convenience function combining the above generate_raw_agg_df and clean_raw_agg_df functions.
def generate_all_artist_df(save_csv=False, verbose=False, rep_stat='median', df_f=df_fa, df_r=df_ra):
    agg_df = generate_raw_agg_df(save_csv, verbose, df_f, df_r)
    df_all = clean_raw_agg_df(agg_df, save_csv, rep_stat)
    return df_all


# Used to create an aggregated track dataframe from the featured (df_ft) and random (df_rt) track dataframes. Using
# the artists within each track's 'artists' column, a valid artist (with no null values) is found within the
# aggregated artist dataframe created by artist_agg. This artist's data is joined to the track's data,
# with any duplicate track ids favoring non-null values found within df_ft. Invalid values are replaced with nulls to
# be later deleted.
def track_agg(group, verbose=False):
    featured_col = 'featured'
    ids_col = 'ids'
    artists_col = 'artists'
    release_date_col = 'release_date'
    release_year_col = 'release_year'
    artist_prefix = 'artist_'

    df_f = df_ft
    df_art_agg = df_a
    track_columns = list(df_f.columns)
    all_track_columns = track_columns + [release_year_col]
    track_data_sets = {}
    track_data_sets_f = {}
    result = {}

    artist_cols = ['featured_count', 'dates', 'ids', 'names', 'first_release', 'last_release', 'num_releases',
                   'num_tracks', 'playlists_found', 'feat_track_ids', 'tracks', 'ambient', 'asian', 'christian',
                   'classical_instrumental', 'contemporary', 'country', 'electronic', 'euro', 'folk', 'hip-hop',
                   'indie_alternative', 'jazz', 'latin', 'metal', 'pop', 'reggae_soul', 'rock',
                   'monthly_listeners_mean', 'monthly_listeners_median', 'monthly_listeners_std_dev',
                   'monthly_listeners_min', 'monthly_listeners_max', 'monthly_listeners_val', 'popularity_mean',
                   'popularity_median', 'popularity_std_dev', 'popularity_min', 'popularity_max', 'popularity_val',
                   'followers_mean', 'followers_median', 'followers_std_dev', 'followers_min', 'followers_max',
                   'followers_val', 'log_monthly_listeners_mean', 'log_monthly_listeners_median',
                   'log_monthly_listeners_std_dev', 'log_monthly_listeners_min', 'log_monthly_listeners_max',
                   'log_monthly_listeners_val']

    track_min_vals = {'featured': 0.0, 'popularity': 0.0, 'markets': 0.0, 'release_date': 0.0,
                      'duration_ms': 0.0, 'acousticness': 0.0, 'danceability': 0.0, 'energy': 0.0,
                      'instrumentalness': 0.0, 'liveness': 0.0, 'loudness': -60.0, 'speechiness': 0.0, 'tempo': 0.0,
                      'valence': 0.0, 'musicalkey': 0.0, 'musicalmode': 0.0, 'time_signature': 0.0, 'count': 0.0}

    track_max_vals = {'featured': 1.0, 'popularity': 100, 'markets': 300.0, 'release_date': datetime.now().year,
                      'duration_ms': 1e7, 'acousticness': 1.0, 'danceability': 1.0, 'energy': 1.0,
                      'instrumentalness': 1.0, 'liveness': 1.0, 'loudness': 5.0, 'speechiness': 1.0, 'tempo': 300.0,
                      'valence': 1.0, 'musicalkey': 12, 'musicalmode': 1.0, 'time_signature': 5.0, 'count': 1e3}

    # Loops through each row within a group to add valid non-null data to track_data_sets (with track_data_sets_f (
    # featured)) acting as a tie-breaker.
    for index, row in group.iterrows():
        for coln in all_track_columns:
            track_data_sets.setdefault(coln, set())
            track_data_sets_f.setdefault(coln, set())
            valid_data = True
            if coln != release_year_col:
                track_data = row[coln]
            else:
                track_data = row[release_date_col]
            if pd.notnull(track_data):
                if coln == release_year_col:
                    try:
                        release_year = datetime.strptime(track_data, '%Y-%m-%d').year
                    except ValueError:
                        try:
                            release_year = datetime.strptime(track_data, '%Y-%m').year
                        except ValueError:
                            try:
                                release_year = datetime.strptime(track_data, '%Y').year
                            except ValueError:
                                release_year = None
                                valid_data = False
                                print(f"\nDate format of release_date {track_data} not parsed.\n")
                    track_data = release_year
                if isinstance(track_data, (int, float, np.integer, np.floating)):
                    if coln in track_min_vals.keys():
                        if track_data < track_min_vals[coln]:
                            valid_data = False
                    if coln in track_max_vals.keys():
                        if track_data > track_max_vals[coln]:
                            valid_data = False
                if valid_data:
                    track_data_sets[coln].add(track_data)
                    if row[featured_col] == 1:
                        track_data_sets_f[coln].add(track_data)

    # Records the data within track_data_sets to the result dictionary (using track_data_sets_f if there's a conflict)
    for cl in all_track_columns:
        if cl in track_data_sets:
            data_set = track_data_sets[cl] if len(track_data_sets[cl]) == 1 else track_data_sets_f[cl]
            result[cl] = list(data_set)[0] if data_set else np.nan

    # Finding a valid artist
    artists_arr = result[artists_col].split(", ") if ", " in result[artists_col] else [result[artists_col]]
    selected_artist, artist_data = None, None
    for artist in artists_arr:
        artist_data = df_art_agg.loc[df_art_agg[ids_col] == artist, artist_cols]
        if not artist_data.empty:
            selected_artist = artist
            break

    # Concatenating (joining) the artist data to the track data
    for cl in artist_cols:
        key = artist_prefix + cl
        if selected_artist:
            imported_data = artist_data[cl].values[0]
        else:
            imported_data = None
        result[key] = imported_data

    if verbose:
        print(result)
    return pd.Series(result)


# Concatenates the featured (df_f=df_ft) and random (df_r=df_rt) track dataframes, creates a new dataframe df_tr_aggr
# by applying track_agg, saves df_tr_aggr to a csv file, then returns it.
def generate_track_agg_df(save_csv=False, verbose=False, df_f=df_ft, df_r=df_rt):
    id_col = 'ids'
    track_agg_save_name = 'raw_all_Spotify_track_info.csv'

    df_trk = pd.concat([df_f, df_r]).reset_index(drop=True)
    df_tr_aggr = df_trk.groupby(by=id_col, as_index=False)[df_trk.columns].apply(lambda g: track_agg(g, verbose))
    if save_csv:
        print(f"\nSaving track aggregate dataframe to {track_agg_save_name}.\n")
        df_tr_aggr.to_csv(track_agg_save_name)
    return df_tr_aggr


# Cleans the aggregated track dataframe by dropping nulls found in null_track_cols_to_drop, fills null statistical
# columns with corresponding values found in their prefix_val columns, saves the cleaned dataframe to a csv file,
# then returns it.
def clean_track_agg_df(df_track_agg, save_csv=False):
    null_track_cols_to_drop = ['popularity', 'release_date', 'duration_ms', 'acousticness', 'danceability', 'energy',
                               'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                               'artist_ids']
    count_col = 'count'
    clean_track_agg_save_name = 'all_Spotify_track_info.csv'

    df_track_agg = df_track_agg.dropna(subset=null_track_cols_to_drop).fillna(value={count_col: 0})
    df_track_agg = df_track_agg.apply(lambda row: fill_nulls(row, list(df_track_agg.columns)), axis=1)
    if save_csv:
        print(f"\nSaving track aggregate dataframe with nulls removed to {clean_track_agg_save_name}.\n")
        df_track_agg.to_csv(clean_track_agg_save_name)
    return df_track_agg


# Convenience function combining the above generate_track_agg_df and clean_track_agg_df functions.
def generate_all_track_df(save_csv=False, verbose=False, df_f=df_ft, df_r=df_rt):
    agg_df = generate_track_agg_df(save_csv, verbose, df_f, df_r)
    df_all = clean_track_agg_df(agg_df, save_csv)
    return agg_df, df_all





# Looking for nulls/duplicates and filling dictionaries. Correcting artists not credited in various tracks.
null_dup_dict = find_nulls_dupes(dictn_df)
fill_uncredited_fast()
# df_fa.groupby(by='ids')[df_fa.columns].apply(lambda g: cross_val_feat_art(g))

df_rt[['ids', 'artists']].apply(lambda row: assign_artists_to_tracks(row['ids'], row['artists']), axis=1)
df_ft[['ids', 'artists']].apply(lambda row: assign_artists_to_tracks(row['ids'], row['artists']), axis=1)
df_fa['feat_track_ids'] = df_fa.apply(lambda row: make_art_f_track_dicts_rmv_dupes(row['dates'], row['ids'],
                                                                                   row['feat_track_ids']), axis=1)
df_fca.apply(lambda row: genre_extract(row['ids'], row['genres'], art_genre_dict_f), axis=1)
df_rca.apply(lambda row: genre_extract(row['ids'], row['genres'], art_genre_dict_r), axis=1)

# only 23 tracks shared between df_ft and df_rt with minor popularity differences - prioritize df_ft version
# for track in dup_tracks:
#    dup_validation(track)


# Creating the clean aggregated artist and track dataframes
df_a = generate_all_artist_df(True, True)
# df_a = pd.read_csv('all_Spotify_artist_info.csv')
df_aggr, df_t = generate_all_track_df(True, True)
# df_t = pd.read_csv('all_Spotify_track_info.csv')








# eof
