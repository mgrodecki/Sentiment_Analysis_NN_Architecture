## Patterned after dr. Gates https://gatesboltonanalytics.com/
## and https://www.kaggle.com/code/ademhph/emotion-recognition-using-lstm
## ######################################################################
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.optimizers import Adam
from keras import layers
from keras import utils
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


## Getting the data loaded and ready
val = pd.read_csv("C:/NeuralNetworks/Final_Project/validation.csv")
train = pd.read_csv("C:/NeuralNetworks/Final_Project/training.csv")
test = pd.read_csv("C:/NeuralNetworks/Final_Project/test.csv")

print("Validation data \n")
print(val.shape)

print("Train data \n")
print(train.shape)

print("Test data \n")
print(test.shape)

## Add validation data to the training corpus (we will use only test data for testing our models)
train = pd.concat([train, val], axis=0)

print("Train data \n")
print(train.shape)

print("The first 20 values of training data: \n", train.head(20))

labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
train['emotion'] = train['label'].map(labels_dict)

print("The first 20 values of training data: \n", train.head(20))

# visualize distribution of different emotion values
print(train.groupby(["emotion","label"]).size())

plt.figure(figsize = (8,8))
plt.bar(labels, train["emotion"].value_counts())

#tokenize
all = train['text'].tolist() + test['text'].tolist()

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(all)
word_index1 = tokenizer1.word_index

#apply stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in word_index1.keys()]
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(stemmed_words)
word_index2 = tokenizer2.word_index

def preprocess_data(data):
    new_data = []
    for index, row in data.iterrows():
        test_split = row['text'].split()
        stemmed_words2 = [stemmer.stem(word) for word in test_split]
        token_list= tokenizer2.texts_to_sequences([stemmed_words2])[0]
        new_data.append([token_list,row['label']]) 
    return new_data

tk_train = preprocess_data(train)
print(train['text'][0])
print(tk_train[0])    

tk_test = preprocess_data(test)
print(test['text'][0])
print(tk_test[0])    


## Set y to the label and x to the text for both test and train datasets
x_train = [row[0] for row in tk_train]
y_train = [row[1] for row in tk_train]

x_test = [row[0] for row in tk_test]
y_test = [row[1] for row in tk_test]


# add padding
length_of_longest_sentence = len(max(x_train, key=len))
print(length_of_longest_sentence)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=length_of_longest_sentence, padding="post")
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=length_of_longest_sentence, padding="post")

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

## What does the data look like?
print("The first value of x_test is \n", x_test[0])
print("The shape of x_test is \n", x_test.shape)
print("The first value of x_train is \n", x_train[0])
print("The shape of x_train is \n", x_train.shape)
print("The first value of y_test is \n", y_test[0])
print("The shape of y_test is \n", y_test.shape)
print("The first value of y_train is \n", y_train[0])
print("The shape of y_train is \n", y_train.shape)
print("The first few values of y_test are \n", y_test[0:10])
print("The first few values of y_train are \n", y_train[0:10])


# Convert labels to one-hot encoding
#y_train = to_categorical(y_train, num_classes=6)
#y_test = to_categorical(y_test, num_classes=6)


NumCols=x_train.shape[1]
print(NumCols)
input_dim = NumCols

NumRows=x_train.shape[0]
print(NumRows)



###############################################
##
## ANN
##
######################################################
ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=NumRows, output_dim=5, input_length=NumCols),
  tf.keras.layers.Dense(50, activation='relu'), 
  tf.keras.layers.Dropout(0.2),   
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(15, activation='relu'), 
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(6, activation='softmax')
])
    
ANN_Model.summary()

ANN_Model.compile(
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Hist=ANN_Model.fit(x_train, y_train, batch_size=12, epochs=5, validation_data=(x_test, y_test))



###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = ANN_Model.evaluate(x_test, y_test)

## Predictions
predictions=ANN_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
print(y_test)
print(predictions)
print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
print(Max_Values)
print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: ANN') 
ax.xaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=0, fontsize = 18)




###############################################
##
## CNN
##
######################################################

CNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=NumRows, output_dim=5, input_length=NumCols),

  tf.keras.layers.Conv1D(kernel_size=3, activation="relu", filters=25), 
  tf.keras.layers.MaxPool1D(pool_size=2), 

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(50, activation='relu'), 
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Dense(6, activation='softmax') 
])


CNN_Model.summary()


CNN_Model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


Hist = CNN_Model.fit(x_train, y_train, batch_size=12, epochs=5, validation_data=(x_test, y_test))


###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = CNN_Model.evaluate(x_test, y_test)

## Predictions
predictions=CNN_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
print(y_test)
print(predictions)
print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
print(Max_Values)
print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: CNN') 
ax.xaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=0, fontsize = 18)




############################################
## LSTM
#############################################
LSTM_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=NumRows, output_dim=5, input_length=NumCols),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40)),
  tf.keras.layers.Dense(6, activation='softmax')
])
    
LSTM_Model.summary()

LSTM_Model.compile(
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )


Hist=LSTM_Model.fit(x_train, y_train, batch_size=12, epochs=5, validation_data=(x_test, y_test))



###### History and Accuracy
plt.figure(figsize = (8,8))
plt.plot(Hist.history['accuracy'], label='accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(Hist.history['loss'], label = 'loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.2])
plt.legend(loc='lower right')



##Test and Model
Test_Loss, Test_Accuracy = LSTM_Model.evaluate(x_test, y_test)

## Predictions
predictions=LSTM_Model.predict([x_test])

## Confusion Matrix and Accuracy - and Visual Options
print("The test accuracy is \n", Test_Accuracy)


print("The prediction accuracy via confusion matrix is:\n")
print(y_test)
print(predictions)
print(predictions.shape)
Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
print(Max_Values)
print(np.argmax([predictions]))
cm = confusion_matrix(Max_Values, y_test)
print(cm)

## Pretty Confusion Matrix
labels = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: LSTM') 
ax.xaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],rotation=0, fontsize = 18)



print("Done!")