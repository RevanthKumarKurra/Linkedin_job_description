import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import unicodedata
import json
import wordcloud

import requests
from bs4 import BeautifulSoup

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer,GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel

import pandas as pd
import numpy as np

from serpapi.google_search import GoogleSearch


import webbrowser

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

def finetuining_llm_model(path=r"C:\linkedin_post\job_data.csv"):
    tokenizer = AutoTokenizer.from_pretrained('t5-large')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-large')

    dataset = load_dataset("csv", data_files=path)
    dataset = dataset.remove_columns(column_names=['Unnamed: 0.1','Unnamed: 0'])
    dataset = dataset['train'].train_test_split(0.1)

    def tokenizing_data(data):
    
        inputs = tokenizer(
            data['Description'],
            max_length=812,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = tokenizer(
            data['Company Name'],
            max_length=15,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        inputs['labels'] = outputs['input_ids']

        return inputs
    
    
    training_data = dataset['train'].map(tokenizing_data,batched=True,batch_size=1)
    training_data = training_data.remove_columns(column_names=['Description','Company Name'])
    print(training_data)

    testing_data = dataset['test'].map(tokenizing_data,batched=True,batch_size=1)
    testing_data = testing_data.remove_columns(column_names=['Description','Company Name'])
    print(testing_data)

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    print(print_number_of_trainable_model_parameters(model))


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=2,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    peft_model_train = get_peft_model(model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model_train))

    #output_dir = "./output_t5_model"

    peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=10,
    no_cuda=True
    )

    peft_trainer = Trainer(
    model=peft_model_train, 
    args=peft_training_args,
    train_dataset = training_data,
    eval_dataset=testing_data,
    )

    #peft_trainer.train()

    #peft_trainer.model.save_pretrained("./output_t5_model")
    #tokenizer.save_pretrained("./output_t5_model")

    

tokenizer_llm = AutoTokenizer.from_pretrained('t5-large')
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
peft_model = PeftModel.from_pretrained(peft_model_base, "./output_t5_model", is_trainable=False)



# Inference function
def predict_company_name(text):
    # Preprocess the text
    text = list(map(cleaning_data, text.split()))
    #print(text)
    text = list(map(preprocessing_text, text))
    text = " ".join(text).strip()
    #print(text)
    input_ids = tokenizer_llm(text,return_tensors="pt").input_ids
    outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    prediction = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
    print(str(prediction))
    return prediction


def get_company_details(company_name):
    
    #company_name = input("Enter the Company name")
    params = {
        "engine": "google",
        "q": "{0} company details".format(company_name),
        "api_key": "cd7b23ff5e4b26a65741935519371546fc9aab509c27faac13d3e3670202ddb4"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get('organic_results',[])[0]['title'],results.get('organic_results',[])[0]['snippet'],results.get('organic_results',[])[1]['displayed_link'].split(sep=" ")[0]



# Route for the main page
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    predicted_company_name_local = "" 
    company_name = ""
    company_desc = ""
    company_link = ""

    if request.method == "POST":
        # Get job description from the form
        job_description = request.form.get("job_description", "")

        # Predict job type
        result = predict_text(job_description)

        if result == "The Post is Genuine":
            # Predict company name
            predicted_company_name_local = predict_company_name(job_description)
            company_name, company_desc, company_link = get_company_details(predicted_company_name_local)

    return render_template(
        "index.html", 
        result=result,
        predicted_company_name=predicted_company_name_local,
        company_name=company_name,
        company_desc=company_desc,
        company_link=company_link
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
