import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import gc
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Download the 'punkt' package
nltk.download('punkt')

# Read a smaller subset of the text
file = open("A-Tale-of-Two-Cities.txt").read()
file = file[:100000]  # Limiting the text to 100,000 characters

# Tokenize the text into words
words = word_tokenize(file)
vocab = sorted(set(words))
word_to_num = {word: i for i, word in enumerate(vocab)}
num_to_word = {i: word for i, word in enumerate(vocab)}
input_len = len(words)
vocab_len = len(vocab)
seq_length = 50
x = []
y = []

for i in range(0, input_len - seq_length, 1):
    in_seq = words[i: i + seq_length]
    out_seq = words[i + seq_length]
    x_to_append = [word_to_num[word] for word in in_seq]
    y_to_append = word_to_num[out_seq]
    x.append(x_to_append)
    y.append(y_to_append)

n_patterns = len(x)

X = np.reshape(x, (n_patterns, seq_length, 1))
X = X / float(vocab_len)

# One hot encoding of the output
y = to_categorical(y, num_classes=vocab_len)

# Clear up memory before model training
gc.collect()

# Building the model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))  # Further reduced units
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Checkpoint to save the model
filepath = "model2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fitting the model with a reduced batch size and epochs
model.fit(X, y, epochs=1, batch_size=32, callbacks=callbacks_list)

# Save the final model
model.save(filepath)

#-----------------------------------------------------------------------------------------
#testing our model(comment the previous code and uncomment the following code)

# import numpy as np
# from keras.models import load_model
# from nltk.tokenize import word_tokenize
#
# # Load the model
# model = load_model("/content/drive/MyDrive/model2.h5")
#
# # Ensure you have the same word-to-num and num-to-word mappings as used during training
# file = open("/content/drive/MyDrive/A-Tale-of-Two-Cities.txt").read()
# words = word_tokenize(file)
# vocab = sorted(set(words))
# word_to_num = {word: i for i, word in enumerate(vocab)}
# num_to_word = {i: word for i, word in enumerate(vocab)}
#
#
# # Function to preprocess input text
# def preprocess_input(text, seq_length):
#     words = word_tokenize(text)
#     # Ensure the input is of the correct sequence length
#     if len(words) < seq_length:
#         words = [''] * (seq_length - len(words)) + words
#     input_seq = [word_to_num.get(word, 0) for word in words[-seq_length:]]  # Use 0 for unknown words
#     return np.reshape(input_seq, (1, seq_length, 1)) / float(len(vocab))
#
#
# # Function to generate text
# def generate_text(model, start_text, num_predict=50, seq_length=50, temperature=1.0):
#     result = []
#     in_text = start_text
#
#     for _ in range(num_predict):
#         # Preprocess the input text
#         X_pred = preprocess_input(in_text, seq_length)
#
#         # Predict the next word
#         prediction = model.predict(X_pred, verbose=0)
#
#         # Apply temperature
#         prediction = np.asarray(prediction).flatten()
#         prediction = np.log(prediction + 1e-7) / temperature
#         prediction = np.exp(prediction) / np.sum(np.exp(prediction))
#
#         # Sample from the probability distribution
#         next_index = np.argmax(np.random.multinomial(1, prediction))
#
#         # Convert the index to a word
#         next_word = num_to_word.get(next_index, '')
#
#         if next_word == '':
#             break
#
#         result.append(next_word)
#
#         # Update the input text
#         in_text = ' '.join(in_text.split()[1:] + [next_word])
#
#     return ' '.join(result)
#
#
# # Example usage
# start_text = "France, less favoured on the whole as to matters spiritual than"
# "her sister of the shield and trident rolled with exceeding"
# "smoothness down hill, making paper money and spending it."
# "Under the guidance of her Christian pastors, she entertained"
# "herself, besides, with such humane achievements as sentencing"
# "a youth to have his hands cut off, his tongue torn out with"
# "pincers, and his body burned alive,"
#
# generated_text = generate_text(model, start_text, num_predict=1, temperature=0.8)
# print(generated_text)

