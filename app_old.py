import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
import json
import wordcloud

import requests
from bs4 import BeautifulSoup

import webbrowser

import keras
import transformers
from transformers import TFT5ForConditionalGeneration,T5Tokenizer
import pickle

from flask import Flask,request,render_template

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Layer,GRU,Bidirectional,Dense
from tensorflow.keras.models import Sequential,Model
from keras.saving import save_model,load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize Flask app
app = Flask(__name__)

# Reading the Text data which contains the Non-Genuine data 
def reading_text(path:'str')->str:
    text_data = []

    with open (path,'r',encoding='utf-8') as text:
        text = text.read()
        posts = text.split("\n\n\n\n")
        for post in posts:
            text_data.append(post.strip())
    text_data = [[item] for item in text_data if len(item) > 0]

    data = pd.DataFrame(text_data,columns=['Job Description'])
    data['Job Type'] = "Not Genuine"

    data.drop_duplicates(inplace=True,ignore_index=True)

    return data

# It is having the "Genuine" jobs data in the JSON format
def reading_json(path:'str') ->str:

    with open(path,'r',encoding='utf-8') as json_file:

        data = json.load(json_file)

    data = pd.DataFrame(data)

    data = pd.DataFrame(data.iloc[0:300],columns=['Description'])

    data['Job Type'] = "Genuine"

    data.rename(columns={'Description':'Job Description'},inplace=True)

    return data

# This is the function that whcih you will get the company description and the company name
def reading_company_data(path:'str') -> str:

    with open(path,'r') as data:
        data = json.load(data)
        data = pd.DataFrame(data)

    return data


text_data = reading_text(r"C:\linkedin_post\hiriring_data.txt")

json_data = reading_json(r"C:\linkedin_post\jobsDetails_dataset.json")

data = pd.concat([text_data,json_data],ignore_index=True)

company_data = reading_company_data(r"C:\linkedin_post\jobsDetails_dataset.json")


## Cleaning the data and removing the unwanted symbols.

ws = WordNetLemmatizer()  # declearing the Lemmatization of word

def cleaning_data(text):

    text = text.lower()
    text = re.sub(r'#', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\-', 'to', text)
    text = re.sub(r'\/', ' ', text)
    text = re.sub(r'\,','',text)
    text = re.sub(r'admin@missionhelpinghands', '', text)
    text = re.sub(r'dr\.shubhra chakraborty', '', text)
    text = re.sub(r"\(",'',text)
    text = re.sub(r"\)",'',text)
    text = re.sub(r'\.',' ',text)
    text = re.sub(r'&','and',text)
    text = re.sub(r"(?<=\d),(?=\d)", '', text)
    text = re.sub(r"(?<=\D),(?=\D)", ' ', text)
    text = re.sub(r'\!',' ',text)
    text = re.sub(r"\|",' ',text)
    text = re.sub(r"\'",'',text)
    text = re.sub(r"\:",' ',text)
    text = re.sub(r'\*','',text)
    text = re.sub(r'\"','',text)
    text = re.sub(r"\+",'',text)
    text = re.sub(r"\?","",text)
    text = re.sub(r"pls","please",text)

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text.lower()  

def preprocessing_text(text):

    text = text.split()
    corpus = [ws.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]# Removing stopwords and lemmitizing the words
    corpus = ' '.join(corpus)

    return corpus    


# Applying the Tokenization(convert into numbers) to the text data

maxlen = 1048 # we are declaring the maxlen Globally

def tokenization(input_data,output_data):
    
    englishword_tokenizer = Tokenizer(oov_token='[UNK]')
    englishword_tokenizer.fit_on_texts(input_data)
    description_seq = englishword_tokenizer.texts_to_sequences(input_data)

    #vocab_size = len(englishword_tokenizer.word_index)+1
    #maxlen = max(len(data) for data in description_seq)

    padded_input_data = pad_sequences(description_seq,maxlen=maxlen,padding='post')

    padded_input_data= np.array(padded_input_data)


    output_data = [[des] for des in output_data ]

    encoder = OneHotEncoder(drop = 'first')
    
    output_data = encoder.fit_transform(output_data).toarray()

    with open(r"C:\linkedin_post\tokenizer.pkl", "wb") as file:
        pickle.dump(englishword_tokenizer, file) # After the tokenization the we will load because each time the tokenizer may have different words so at each time we are loading it.


    return padded_input_data,output_data

#input_data,output_data,vocab_size,englishword_tokenizer,maxlen = tokenization(input_data=data['Job Description'].to_list(),output_data=data['Job Type'].to_list())




with open(r"C:\linkedin_post\tokenizer.pkl", "rb") as file:  # loading the tokenizier that which is pickle file to the tokenizer
        tokenizer = pickle.load(file)


## This Function is for training the model we cannot train model each time so I created it in the form of the function when you function call you can train the data
def training_model(input_data,output_data):

    epochs = int(input("Enter the No.of Epochs:"))
    batch_size = int(input("Enter the No.of Batch Size:"))

    units = 512

    vocab_size = len(tokenizer.word_index)+1

    model = Sequential(
        [
            Embedding(input_dim = vocab_size,output_dim=units,input_length = maxlen),
            Bidirectional(GRU(units)),
            Dense(2,activation='softmax')
        ] )

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(input_data,output_data,epochs=epochs,batch_size=batch_size)

    save_model(model,"C:\\linkedin_post\\model.h5") # The model after training it will be save the weights

# This function is for the training the model
def training_data(data):

    data['Job Description'] = data['Job Description'].map(lambda data:cleaning_data(data))
    data['Job Description'] = data['Job Description'].map(lambda data:preprocessing_text(data))

    input_data,output_data = tokenization(input_data=data['Job Description'].to_list(),output_data=data['Job Type'].to_list())

    training_model(input_data=input_data,output_data=output_data)


#training_model(input_data=input_data,output_data=output_data)

# the Model is loaded to predict the linkedin post
model = load_model(r"C:\linkedin_post\model.h5")

# This is the Function for the prediction of the post
def predict_text(text):
    # Preprocess the text
    text = list(map(cleaning_data, text.split()))
    text = list(map(preprocessing_text, text))
    text = " ".join(text).strip().split()

    text_l = []
    for word in text:
        if word not in tokenizer.word_index:
            text_l.append(tokenizer.word_index['[UNK]'])
        else:
            text_l.append(tokenizer.word_index[word])
    
    text_l = pad_sequences([text_l], maxlen=maxlen, padding='post')
    text_l = np.array(text_l)

    s = model.predict(text_l)

    if round(s[0][0]) == 1:
        
        return "The Post is Genuine"
    
    else:

        return "The Post is Not Genuine"

# This is the ploted graph that which we will plot to see the most frequent words in the form of wordcloud.
def plot_graph(data):

    genuine_job = ' '.join(data[data['Job Type'] == 'Genuine']['Job Description'])
    wordcloud_genuine = wordcloud.WordCloud(width=800,height=400,background_color='black').generate(genuine_job)
    img1 = plt.imshow(wordcloud_genuine,interpolation='bilinear')
    img1= plt.axis('off')
    img1 = plt.show()

    nongenuine_job = ' '.join(data[data['Job Type'] == 'Not Genuine']['Job Description'])
    wordcloud_nongenuine = wordcloud.WordCloud(width=800,height=400,background_color='black').generate(nongenuine_job)
    img2 = plt.imshow(wordcloud_nongenuine,interpolation='bilinear')
    img2 = plt.axis('off')
    img2 = plt.show()

    return img1,img2


"""The code for the company name from the job description from the pretrained 'T5' LLM model"""

# The model and the tokenizer is took from the pretrained T5-small llm (encoder-decoder) model 
tokenizer_comp = T5Tokenizer.from_pretrained('t5-small')
model_comp = TFT5ForConditionalGeneration.from_pretrained('t5-small')

des_maxlen = 1480
com_maxlen = 16

# This function is for prepraining the training data from the LLM.
def training_data_llm(data):
    # Preprocess the data
    data['Description'] = data['Description'].map(lambda x: preprocessing_text(cleaning_data(x)))
    data['Company Name'] = data['Company Name'].map(lambda x: preprocessing_text(cleaning_data(x)))

    # Tokenization with fixed maximum lengths
    input_tokenized_data = tokenizer_comp(
        data['Description'].tolist(),
        max_length=des_maxlen,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    ).input_ids

    output_tokenized_data = tokenizer_comp(
        data['Company Name'].tolist(),
        max_length=com_maxlen,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    ).input_ids

    # Prepare shifted decoder input IDs for teacher forcing
    decoder_input_ids = []
    for tokens in output_tokenized_data:
        shifted = tf.concat([tf.constant([tokenizer.pad_token_id]), tokens[:-1]], axis=0)
        decoder_input_ids.append(shifted)
    decoder_input_ids = tf.convert_to_tensor(decoder_input_ids)

    return input_tokenized_data, output_tokenized_data, decoder_input_ids


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training function
@tf.function
def train_step(input_ids, decoder_input_ids, labels, model):
    with tf.GradientTape() as tape:
        outputs = model_comp(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs.loss

    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
def train_model(data, tokenizer, model):
    input_tokenized_data, output_tokenized_data, decoder_input_ids = training_data_llm(data, tokenizer, model)
    epochs = int(input("Enter the Number of Epochs"))
    batch_size = int(input("Enter the Batch Size"))
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(data), batch_size):
            # Prepare batches
            batch_input_ids = input_tokenized_data[i:i + batch_size]
            batch_decoder_input_ids = decoder_input_ids[i:i + batch_size]
            batch_labels = output_tokenized_data[i:i + batch_size]

            # Train on batch
            loss = train_step(batch_input_ids, batch_decoder_input_ids, batch_labels, model)
            if i % 50 == 0:
                print(f"Batch {i}/{len(data)} - Loss: {loss.numpy()}")

            model.save_pretrained(r"c:\linkedin_post\t5_model")

model_pre = TFT5ForConditionalGeneration.from_pretrained(r"C:\linkedin_post\t5_model")

# Inference function
def predict_company_name(text):
    # Preprocess the text
    text = list(map(cleaning_data, text.split()))
    text = list(map(preprocessing_text, text))
    text = " ".join(text).strip().split()
    input_ids = tokenizer_comp.encode(text, return_tensors='tf', truncation=True, max_length=1506)
    outputs = model_pre.generate(input_ids, max_length=16)
    prediction = tokenizer_comp.decode(outputs[0], skip_special_tokens=True)
    return prediction


def get_full_company_name(company_name):
    try:
        # Replace spaces with '+' for the URL
        search_query = company_name + "company name as their websites"
        url = f"https://www.google.com/search?q={search_query}"
        
        # Send a GET request to Google Search
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code != 200:
            return "Error: Unable to fetch search results."

        # Parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the first result from the search results
        result = soup.find("h3")
        
        if result:
            text = result.text
            
            text = text.lower()
            text = re.sub('wikipedia','',text)
            text = re.sub('-','',text)
            text = re.sub('linkedin','',text)
            text = re.sub('about us','',text)
            text = re.sub("/|",'',text)
            text = re.sub("official website",'',text)
            
            return text.strip()
        
        else:
            
            return "Full company name not found."
    
    except Exception as e:
        return f"Error: {str(e)}"


## This is the Function that which we can get link check wheather company is listed or not

def get_company_link(name):
    url = f"https://www.quickcompany.in/company/search?q={name}"
    webbrowser.open(url)
    print(f"The URL {url} has been opened in your default web browser.")

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    company_name = ""
    company_link = ""
    if request.method == "POST":
        # Get job description from the form
        job_description = request.form.get("job_description", "")

        # Predict job type
        result = predict_text(job_description)

        if result == "The Post is Genuine":
            # Predict company name
            predicted_company_name = predict_company_name(job_description)
            company_name = get_full_company_name(predicted_company_name)
            company_link = f"https://www.quickcompany.in/company/search?q={company_name}"
    
    return render_template(
        "index.html", 
        result=result, 
        company_name=company_name,
        company_link=company_link)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
