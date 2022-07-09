
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from numpy.random import rand
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from transformers import RobertaTokenizer, TFRobertaForMaskedLM


def model_download():
    import tarfile
    try:
        #!wget https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/models/indic-bert-v1.tar.gz
        file = tarfile.open('indic-bert-v1.tar.gz')
    except:
        pass
    file.extractall('./ashish')
    file.close()
def load_model_t(model_name):
    if model_name=='xmlr':
        from transformers import XLMRobertaTokenizer,TFXLMRobertaForSequenceClassification,XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        models = TFXLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',num_labels=4,from_pt=True)
        #roberta-base

    elif model_name=='mbert':
        from transformers import BertTokenizer, TFBertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        models =TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=4)

    elif model_name=='indicbert':
        #model_download()
        from transformers import AlbertConfig, TFAlbertModel,TFAlbertForSequenceClassification,AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained('ai4bharat/indic-bert')
        models = TFAlbertForSequenceClassification.from_pretrained('ai4bharat/indic-bert',num_labels=4,from_pt=True)

    elif model_name=='sanbert':
        from transformers import AlbertConfig, AlbertTokenizer,TFAlbertForSequenceClassification
        tokenizer = AlbertTokenizer.from_pretrained('/home/kabira/Downloads/data_dir')
        models = TFAlbertForSequenceClassification.from_pretrained('/home/kabira/Downloads/checkpoints/checkpoint-244000',from_pt=True,num_labels=4)
    else:
        print("please select a valid model")
    print("model loaded sucessfully")

    return tokenizer,models


import pandas as pd
maplab = ["Avyayibhava","Bahuvrihi","Dvandva","Tatpurusha"]

def data_preps(task,mp,pp):
    if pp=='train':
        data = pd.read_csv('data/train_large.csv')
        sentences=data['Context']
        labels=data['labels']
        
        sent = data["Compounds"]
    if pp=="test":
        test = pd.read_csv('data/test_large.csv')
        sentences = test['Context']
        labels = test['labels']
        sent = test["Compounds"]
        print("length of ",len(test))
    if pp=="dev":
        test = pd.read_csv('data/dev_large.csv')
        sentences = test['Context']
        labels = test['labels']
        sent = test["Compounds"]
        print("length of ",len(test))
    
    labels=[maplab.index(labels[i]) for i in range(len(labels))]


    if task=='compounds':
        new_d = pd.DataFrame({"class":sent,"labels":labels})
        data = new_d.drop_duplicates()
        sentences,labels = data['class'],data['labels']
    elif task=='context':
        #sentences = sent+" [SEP] "+sentences
        sentences = sent+" "+sentences
        labels=labels
    elif task=='slp1':
        if mp=='context':
            sentences = sent+" [SEP] "+sentences
        elif mp=='compounds':
            new_d = pd.DataFrame({"class":sent,"labels":labels})
            data = new_d.drop_duplicates().reset_index()
            sentences,labels = data['class'],data['labels']
        else:
            print("Please select a cmp in slp1")

        sent_slp1=[]
        for i in range(len(sentences)):
            text1 = transliterate(sentences[i],sanscript.DEVANAGARI, sanscript.SLP1)
            sent_slp1.append(text1)
        sentences = sent_slp1       
    else:
        print("PLEASE SELECT A TASK")
    # print("halka ",sentences)
    return sentences,labels

def data_model_preps(max_length,tokenizer,model,sentences_train,labels_train,sentences_dev,labels_dev,model_save_path):
    input_ids=[]
    attention_masks=[]
    for sent in sentences_train:
        bert_inp=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =max_length,pad_to_max_length = True,return_attention_mask = True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids_train =np.asarray(input_ids)
    attention_masks_train =np.array(attention_masks)
    labels_train = np.array(labels_train)
    print('Train inp shape {} Train label shape {} Train attention mask shape {} '.format(input_ids_train.shape,labels_train.shape,attention_masks_train.shape))

    input_ids=[]
    attention_masks=[]
    for sent in sentences_dev:
        bert_inp=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =max_length,pad_to_max_length = True,return_attention_mask = True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids_dev =np.asarray(input_ids)
    attention_masks_dev =np.array(attention_masks)
    labels_dev = np.array(labels_dev)
    print('Dev inp shape {} Dev label shape {} dev attention mask shape {} '.format(input_ids_dev.shape,labels_dev.shape,attention_masks_dev.shape))

    #Save model
    print('**********************************************************************************************************')

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    # model_save_path='/NLP'
    log_dir='tensorboard_data/tb_bert'
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]
    print('\nBert Model',model.summary())
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-8)
    model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

    return model,input_ids_train,input_ids_dev,labels_train,labels_dev,attention_masks_train,attention_masks_dev,callbacks


def train(tokenizer,model,batch_size,epochs,train_inp,val_inp,train_label,val_label,train_mask,val_mask,callbacks,model_save_path):
    model.fit([train_inp,train_mask],train_label,batch_size=batch_size,epochs=epochs,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)
    # tokenizer.save_pretrained(model_save_path)
    # model.save_pretrained(model_save_path)
    return model,tokenizer

def test_results(tokenizer,model,max_length,checkpoint_filepath,task,mp,file_which):

    sentences_test,labelst = data_preps(task,mp,file_which)
    print("test data looks like ",sentences_test[0],labelst[0])
    input_ids_test=[]
    attention_masks_test=[]
    model.load_weights(checkpoint_filepath)

    for sent in sentences_test:
        bert_inp=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =max_length,pad_to_max_length = True,return_attention_mask = True)
        input_ids_test.append(bert_inp['input_ids'])
        attention_masks_test.append(bert_inp['attention_mask'])

    input_ids_test=np.asarray(input_ids_test)
    attention_masks_test=np.array(attention_masks_test)
    labelst=np.array(labelst)
    preds = model.predict([input_ids_test,attention_masks_test])
    lis = np.argmax(preds[0],axis=1)
    target_names = ["Avyayibhava","Bahuvrihi","Dvandva","Tatpurusha"]
    import pandas as pd
    df = pd.DataFrame({"sent":sentences_test,"lables":labelst,"preds":lis})
    df.to_csv(file_which+"prediction_best_models.csv")
    print("files saved success")
    print(classification_report(labelst, lis, target_names=target_names,digits=5))
    return str(classification_report(labelst, lis, target_names=target_names,digits=5))

