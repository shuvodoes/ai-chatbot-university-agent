from flask import Flask, request, jsonify, send_from_directory
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load necessary files
limit = WordNetLemmatizer()
data_file = open('data_ius.json').read()
intents = json.loads(data_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Function to clean and lemmatize the sentence
def clean(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [limit.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    sentence_words = clean(sentence)
    word_counts = Counter(sentence_words)
    bag = [word_counts.get(word, 0) for word in words]
    return np.array(bag)

# Function to predict the intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_TRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get the response from the intent
def get_response(intents_list, intents_json):
    result = "Sorry, I didn't understand that."
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result

# Serve the HTML page for the chatbot
@app.route('/')
def serve_static_html():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')

# Handle chat messages from the frontend
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    intents_list = predict_class(user_message)  # Get the list of predicted intents
    response = get_response(intents_list, intents)  # Pass intents_list to get_response
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
