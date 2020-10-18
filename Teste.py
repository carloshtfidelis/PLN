import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as sk

from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tratarData(path = "C:/Users/carlo/Desktop/USP/Semestre 12/PLN/EP1/ep1_abortion_train.csv"):

    data = pd.read_csv(path, sep = ';', encoding='latin-1')

    sentences = np.array(data['text'])
    labels = np.array(data['abortion'])

    np.place(labels, labels == 'for',"-1")
    np.place(labels, labels == 'neutral',"0")
    np.place(labels, labels == 'against',"1")

    return sentences, labels

sentences, labels = tratarData("C:/Users/carlo/Desktop/USP/Semestre 12/PLN/EP1/ep1_abortion_train.csv")
vocab_size = 3000
embedding_dim = 16
num_epochs = 500
kf = KFold(10, shuffle = True, random_state = 42)

for train, test in kf.split(sentences):
    train_sentences = sentences[train]   
    test_sentences = sentences[test]    

    tokenizer = Tokenizer(num_words = vocab_size  , oov_token="<OOV>")
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences)

    train_padded = np.array(train_padded, dtype=np.float32)
    train_labels = np.array(labels[train], dtype=np.float32)
    test_padded = np.array(test_padded, dtype=np.float32)
    test_labels = np.array(labels[test], dtype=np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

    history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2)
    
