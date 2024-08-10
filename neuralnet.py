import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import random as tf_random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

# Adapted from Dr. Festus Elleh's “D213 Task 2 Data Preprocessing in Python” and “D213 Task 2 Building NN Model in Python” lecture videos, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8639374a-964b-4ae9-b33b-b1210052c07d
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1b9aff54-735f-456a-a6b4-b11b00eb8d2f

# Set to true for greater output (takes longer to run)
verbose = False

df_amz = pd.read_csv(
    'data/D213-sentiment_labelled_sentences-files/sentiment labelled sentences/amazon_cells_labelled.txt',
    delimiter='\t', header=None, names=['review', 'sentiment'])
df_amz['source'] = 'Amazon'
df_imdb = pd.read_csv('data/D213-sentiment_labelled_sentences-files/sentiment labelled sentences/imdb_labelled.txt',
                      delimiter='\t', header=None, names=['review', 'sentiment'])
df_imdb['source'] = 'IMDB'
df_yelp = pd.read_csv('data/D213-sentiment_labelled_sentences-files/sentiment labelled sentences/yelp_labelled.txt',
                      delimiter='\t', header=None, names=['review', 'sentiment'])
df_yelp['source'] = 'Yelp'

df = pd.concat([df_amz, df_imdb, df_yelp], ignore_index=True)
print(df)

# Verifying dataframe has no null values
df.isnull().sum()

if verbose:
    plt.hist(df['sentiment'])
    plt.show()


# Creates a list of unique characters 'list_of_chars' and list of unique words 'word_bag' from input 'reviews'
def char_word_bags(reviews=df['review'], display=False):
    list_of_chars = []
    word_bag = []
    for comment in reviews:
        for character in comment:
            if character not in list_of_chars:
                list_of_chars.append(character)
    if display:
        print(list_of_chars)

    for revw in reviews:
        for word in revw.split(" "):
            if word not in word_bag:
                word_bag.append(word)
    if display:
        print(word_bag)

    return list_of_chars, word_bag


if verbose:
    raw_char_list, raw_word_bag = char_word_bags(df['review'], False)
    print(f"\nList of unique characters (before processing):\n{raw_char_list}\n")
    print(f"\nList of unique words (before processing):\n{raw_word_bag}\n")

stop_words = stopwords.words('english')
description_list = []
raw_reviews = []
for description in df['review']:
    raw_reviews.append(description)
    # remove punctuation, special characters, single characters, and multiple spaces
    description = re.sub("[^a-zA-Z]", " ", description)
    description = re.sub("\s+[a-zA-Z]\s+", " ", description)
    description = re.sub("\s+", " ", description)

    # convert to lower case
    description = description.lower()

    # tokenization
    description = word_tokenize(description)

    # lemmatization
    lemma = WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]

    # removing stop words
    description = [word for word in description if not word in stop_words]
    description = " ".join(description)
    description_list.append(description)

if verbose:
    print(f"\nStop words:\n{stop_words}\n")
    print(f"\nOriginal reviews:\n{raw_reviews}\n")
    print(f"\nCleaned reviews:\n{description_list}\n")
    clean_char_list, clean_word_bag = char_word_bags(description_list, False)
    print(f"List of unique characters (after processing):\n{clean_char_list}\n")
    print(f"Length of above list (expecting 26 letters and 1 space): {len(clean_char_list)}\n")
    print(f"List of unique words (after processing):\n{clean_word_bag}\n")

# Tokenizing
tokens = Tokenizer()
tokens.fit_on_texts(df['review'])
vocab_size = len(tokens.word_index) + 1
max_sequence_embedding = int(np.sqrt(np.sqrt(vocab_size)) // 1 + 1)

print(f"Vocabulary size: {vocab_size}\n")
print(f"Embedding length (vocab_size^(1/4) (rounded up to nearest integer)): {max_sequence_embedding}\n")
if verbose:
    print(f"Word index:\n{tokens.word_index}\n")

# Finding min, median, mean, and max lengths of reviews. Also creating arrays for reviews of length over long_rev_thresh.
long_rev_thresh = 50
long_rev_text = []
review_length = []
for rev in df['review']:
    rev_length = len(rev.split(" "))
    review_length.append(rev_length)
    if rev_length > long_rev_thresh:
        long_rev_text.append(rev)

review_max = np.max(review_length)
review_mean = np.mean(review_length)
review_median = np.median(review_length)
review_min = np.min(review_length)
print(f"Maximum review length (words): {review_max}\n")
print(f"Mean review length (words): {round(review_mean, 2)}\n")
print(f"Median review length (words): {review_median}\n")
print(f"Minimum review length (words): {review_min}\n")

if verbose:
    hist_review_lengths = np.array(review_length)
    print(f"There are {len(long_rev_text)} reviews above {long_rev_thresh} characters:\n{long_rev_text}\n")
    plt.hist(hist_review_lengths[hist_review_lengths >= long_rev_thresh])
    plt.xlim(long_rev_thresh, review_max + 1)
    plt.xlabel('Review length (words)')
    plt.title(f'Distribution of reviews with length over {long_rev_thresh}')
    plt.show()
    plt.hist(hist_review_lengths[hist_review_lengths < long_rev_thresh], bins=long_rev_thresh)
    plt.xlim(0, long_rev_thresh + 1)
    plt.xlabel('Review length (words)')
    plt.title(f'Distribution of reviews with length below {long_rev_thresh}')
    plt.show()


# Splits data into training and test sets then pads according to 'max_padding_length', with padding and truncating both occuring 'post'.
def pad_split(max_padding_length, save=False):
    X = np.array(description_list)
    y = df['sentiment'].values

    # Setting random seeds
    rd = 40
    tf_random.set_seed(16)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rd, stratify=y)

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    print(f"Training size and testing size: {X_train.shape}, {X_test.shape}\n")

    max_length_pad = max_padding_length
    padding_type = 'post'
    trunc_type = 'post'

    sequences_train = tokens.texts_to_sequences(X_train)
    padded_train = pad_sequences(sequences_train, maxlen=max_length_pad, padding=padding_type, truncating=trunc_type)

    sequences_test = tokens.texts_to_sequences(X_test)
    padded_test = pad_sequences(sequences_test, maxlen=max_length_pad, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(padded_train)
    training_label = np.array(y_train)
    test_padded = np.array(padded_test)
    test_label = np.array(y_test)

    if save:
        pd.DataFrame(training_padded).to_csv("training_padded.csv")
        pd.DataFrame(training_label).to_csv("training_label.csv")
        pd.DataFrame(test_padded).to_csv("test_padded.csv")
        pd.DataFrame(test_label).to_csv("test_label.csv")

    return X_train, X_test, y_train, y_test, training_padded, training_label, test_padded, test_label


X_train, X_test, y_train, y_test, training_padded, training_label, test_padded, test_label = pad_split(50, False)

if verbose:
    print(training_padded[15])


# Creates RNN with an embedding layer determined by 'vocabulary' and 'embedding_dim', then uses a GlobalAveragePooling1D layer, and two dense layers.
# The first dense layer has nodes 'nodes' (using 'relu') and the second uses the activation function 'activation'.
def model_creation(nodes=128, vocabulary=vocab_size, embedding_dim=max_sequence_embedding, activation='sigmoid'):
    model = Sequential([
        Embedding(input_dim=vocabulary, output_dim=embedding_dim),
        GlobalAveragePooling1D(),
        Dense(nodes, activation='relu'),
        Dense(2, activation=activation)
    ])
    return model


# Creates RNN with an embedding layer determined by 'vocabulary' and 'embedding_dim', then uses a GlobalAveragePooling1D layer, and three dense layers.
# The first dense layer has nodes 'nodes', the second has nodes 'nodes//2' (both use 'relu'), and the third uses the activation function 'activation'.
def model_creation_3Dense(nodes=128, vocabulary=vocab_size, embedding_dim=max_sequence_embedding, activation='sigmoid'):
    model = Sequential([
        Embedding(input_dim=vocabulary, output_dim=embedding_dim),
        GlobalAveragePooling1D(),
        Dense(nodes, activation='relu'),
        Dense(nodes // 2, activation='relu'),
        Dense(2, activation=activation)
    ])
    return model


# Plot the metric (such as accuracy or loss) and its validation score vs epoch
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()
    return


# Compiles and fits RNN model and provides model evaluation/summary measures on training and test data with predictions
def model_eval(model, train_pad=training_padded, train_labels=training_label, test_pad=test_padded,
               test_labels=test_label, X_tst=X_test, y_tst=y_test,
               loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], batch_size=16,
               num_epochs=20, patience=10):
    validation_splitting = 0.3
    shfl = True
    vrbs = True
    test_index = np.random.randint(0, len(X_tst))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(train_pad, train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_split=validation_splitting, shuffle=shfl,
                        callbacks=[EarlyStopping(patience=patience)], verbose=vrbs)
    print(model.summary())

    predictions_tr = model.predict(train_pad)
    predictions_te = model.predict(test_pad)

    print(f"\nText of review at index {test_index} to be predicted:\n{X_tst[test_index]}\n")
    print(f"Predicted: {'Negative' if predictions_te[test_index][0] >= 0.5 else 'Positive'} review\n")
    print(f"Actual: {'Negative' if y_tst[test_index] == 0 else 'Positive'} review\n")

    score_tr = model.evaluate(train_pad, train_labels, verbose=0)
    print(f"Training loss: {score_tr[0]} / Training accuracy: {score_tr[1]}")

    score_te = model.evaluate(test_pad, test_labels, verbose=0)
    print(f"Test loss: {score_te[0]} / Test accuracy: {score_te[1]}")

    predictions_te_converted = np.apply_along_axis(lambda x: 0 if x[0] >= 0.4999999 else 1, 1, predictions_te)
    cnfs_mtx = confusion_matrix(test_labels, predictions_te_converted)
    print(f"\nConfusion matrix on test data vs predictions:\n{cnfs_mtx}\n")

    plot_graphs(history, metrics[0])
    plot_graphs(history, "loss")
    return


# X_train, X_test, y_train, y_test, training_padded, training_label, test_padded, test_label = pad_split(50, False)
model0 = model_creation()
model_eval(model0)


