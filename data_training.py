import json
import random
import pickle
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
data_file = open('data_ius.json').read()
intents = json.loads(data_file)


words = []
classes = []
documents = []


for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = re.sub(r'[^\w\s]', '', pattern.lower())
        word_list = nltk.word_tokenize(pattern)
        documents.append((word_list, intent['tag']))
        for word in word_list:
            word = lemmatizer.lemmatize(word)
            words.append(word)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(classes).reshape(-1, 1))

training_bag = []
training_output_row = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower())for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = encoder.transform([[document[1]]])
    # Append to the training lists
    training_bag.append(bag)
    training_output_row.append(output_row.flatten())  # Flatten to 1D array

# Convert to NumPy arrays
training_bag = np.array(training_bag)
training_output_row = np.array(training_output_row)


from sklearn.utils import shuffle

train_x, train_y = shuffle(training_bag, training_output_row, random_state=0)


model =  Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)
print("Training done!")


