Name: Ahmad Hirbawy
Email: ahmadhir33@gmail.com

Overview
This project is a machine learning application built using Python, designed to predict the next word in a text sequence. 
The core of the project involves training an LSTM (Long Short-Term Memory) neural network on the classic novel: 
"A Tale of Two Cities".

Functionality
Data Preparation:

Text Reading and Tokenization:
Reads a subset of A Tale of Two Cities (limited to 100,000 characters) from a text file.
Tokenizes the text into individual words using NLTK’s word_tokenize.

Vocabulary Creation:
Creates a vocabulary of unique words from the tokenized text.
Maps words to numerical indices and vice versa for model processing.

Sequence Creation:
Constructs input sequences of length seq_length (50 words) and their corresponding target words.
Converts sequences to numerical format and reshapes them for LSTM input.
One-hot encodes the target words to prepare them for classification.

Model Building:

Architecture:
LSTM Layers:
Three LSTM layers are stacked, with 512 units in the first two layers and 256 units in the final layer. The first two 
LSTM layers return sequences, while the final LSTM layer outputs the prediction.

Dense Layer:
A dense layer with a softmax activation function is used to output probabilities for the vocabulary words.

Compilation:
Compiles the model using the Adam optimizer and categorical cross-entropy loss function, suitable for multi-class classification tasks.

Model Training:
Trains the model with a batch size of 32 and for a single epoch, using the prepared sequences as input and one-hot encoded target words.

Final Model Saving:
Saves the trained model to a file named model2.h5.

Note: if you want to test the model, then comments the first part of the code and uncomment the second part.
