#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import numpy as np


# In[1]:


with open("train_qa.txt" ,"rb") as fp:
    train_data = pickle.load(fp)
    # Unpickle the data 


# In[3]:


train_data


# In[4]:


with open("test_qa.txt" ,"rb") as fp:
    test_data = pickle.load(fp)


# In[5]:


test_data


# In[6]:


type(test_data)
type(train_data)


# In[7]:


#LENGTH OF DATA SETS
len(test_data)


# In[8]:


len(train_data)


# In[9]:


train_data[0]


# In[10]:


#For seeing only the story 
' '.join(train_data[0][0])


# In[11]:


#For seeing only the question 
' '.join(train_data[0][1])


# In[12]:


#For only answer of the ques
train_data[0][2]


# In[13]:


#Setting up the vocabulary
vocab = set()


# In[14]:


all_data = test_data + train_data


# In[15]:


all_data


# In[16]:


for story, question ,answers in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


# In[17]:


vocab.add('yes')
vocab.add('no')


# In[18]:


len(vocab)


# In[19]:


vocab


# In[20]:


# as we need one more space for keras substn
vocab_length=len(vocab)+1


# In[21]:


# max length of story and ques
for data in all_data:
    print(data[0])
    print("\n")
    print(len(data[0]))


# In[22]:


max_story_len = max([len(data[0]) for data in all_data])
max_story_len


# In[23]:


max_ques_len = max([len(data[1]) for data in all_data])
max_ques_len


# In[24]:


# Vectorise the data i.e converrt the data into numerical form using keras


# In[25]:


from tensorflow import keras
from keras.preprocessing.text import Tokenizer


# In[26]:


import keras.preprocessing.sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[27]:


tokenizer = Tokenizer(filters = [])


# In[28]:


#updates internal library based on the texts
tokenizer.fit_on_texts(vocab)


# In[29]:


# Respective id's has been given to the words in vocab
tokenizer.word_index


# In[30]:


#Append story and questions from training data set
train_story_text = []
train_question_text = []
train_answer_text = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answer_text.append(answer)
    


# In[31]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)


# In[32]:


train_story_seq


# In[33]:


#Vectorize the data by making a function
def vector_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_ques_len=max_ques_len):
#Data- Consists of the data like story,ques etc
#Word index = From Vocabaulary the assigned digits to the words
#Max_story/ques_length is for pad_sequences
    X=[] # Stories
    Xq=[] #Question/Queries
    Y=[] # Answers/Output
    for story,question,answer in data:
        x=[word_index[word.lower()] for word in story] # to get word index of each word
        xq=[word_index[word.lower()] for word in question]
        # Gives zeroes and one's for answers
        y=np.zeros(len(word_index)+1) 
        y[word_index[answer]]=1 

        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return(pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_ques_len),
           np.array(Y))
        


# In[34]:


inputs_train,queries_train,answers_train = vector_stories(train_data)


# In[35]:


inputs_test,queries_test,answers_test = vector_stories(test_data)


# In[36]:


inputs_train


# In[37]:


queries_train


# In[38]:


answers_train


# In[39]:


tokenizer.word_index['yes']


# In[40]:


tokenizer.word_index['no']


# In[41]:


#PREPROCESSING


# In[42]:


from keras.models import Sequential, Model
from keras.layers import Embedding
from keras.layers import Input,Activation,Dense,Dropout,Permute,add,dot,concatenate,LSTM


# In[43]:


input_sequence = Input((max_story_len,))


# In[44]:


question = Input((max_ques_len,))


# In[45]:


#Create Objects
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_length, output_dim = 64))
input_encoder_m.add(Dropout(0.3))


# In[46]:


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_length, output_dim = max_ques_len))
input_encoder_c.add(Dropout(0.3))


# In[47]:


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_length, output_dim = 64,input_length=max_ques_len))
question_encoder.add(Dropout(0.3))


# In[48]:


input_encoded_m  = input_encoder_m(input_sequence)
input_encoded_c  = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# In[49]:


match = dot([input_encoded_m ,question_encoded], axes=(2,2))
match = Activation('softmax')(match)


# In[50]:


response = add([match,input_encoded_c])
response = Permute((2,1))(response)


# In[51]:


answer = concatenate([response,question_encoded])


# In[52]:


answer


# In[53]:


answer = LSTM(32)(answer)


# In[54]:


answer = Dropout(0.5)(answer)
answer = Dense(vocab_length)(answer)


# In[55]:


answer = Activation('softmax')(answer)


# In[56]:


model = Model([input_sequence,question],answer)
model.compile(optimizer='rmsprop',loss ='categorical_crossentropy',metrics=['accuracy'])


# In[57]:


model.summary()


# In[58]:


history = model.fit([inputs_train,queries_train],answers_train,
                batch_size= 32, epochs = 30,
                   validation_data=([inputs_test , queries_test] , answers_test)
                        )


# In[59]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])


# In[60]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")


# In[63]:


#SAve
model.save("CHatBot_MOdel")


# In[64]:


#EValuation of test set
model.load_weights("CHatBot_MOdel")


# In[65]:


pred_res = model.predict(([inputs_test,queries_test]))


# In[66]:


test_data[0][0]


# In[71]:


story=' '.join(word for word in test_data[0][0])


# In[72]:


story


# In[73]:


query =' '.join(word for word in test_data[0][1])


# In[74]:


query


# In[79]:


# Prediction MOdel for specified query like in this case it is zero 
val_max = np.argmax(pred_res[0])

for key , val in tokenizer.word_index.items():
    if val==val_max:
        k=key
        
print("Predicted answer is :",k)
print("Probability of certainity is",pred_res[0][val_max])


# In[152]:


vocab


# In[162]:


story = "Mary picked the apple . Sandra went to office . "
story.split()


# In[179]:


my_question = "Is Sandra in the office ? "


# In[180]:


my_question.split()


# In[181]:


my_data = [(story.split(),my_question.split(),'yes')]


# In[182]:


my_story , my_question , my_ans = vector_stories(my_data)


# In[187]:


my_story=' '.join(word for word in test_data[0][0])


# In[188]:


my_question =' '.join(word for word in test_data[0][1])


# In[190]:


my_question


# In[186]:


val_max = np.argmax(pred_res[0])

for key , val in tokenizer.word_index.items():
    if val==val_max:
        k=key
        
print("Predicted answer is :",k)
print("Probability of certainity is",pred_res[0][val_max])


# In[ ]:





# In[ ]:




