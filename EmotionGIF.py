
# coding: utf-8

# In[1]:


import pandas as pd
import re
from random import random
import emoji
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from collections import Counter, defaultdict
import torch
from torch.nn import BCEWithLogitsLoss, MultiLabelMarginLoss
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[2]:


from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers.modeling_bert import BertPreTrainedModel


# In[29]:


df = pd.read_json('./source/train_gold.json', lines=True)


# ## Data cleaning

# In[30]:


# https://github.com/minerva-ml/open-solution-toxic-comments/blob/master/external_data/apostrophes.json
apostrophes = {
  "arent": "are not",
  "cant": "cannot",
  "couldnt": "could not",
  "didnt": "did not",
  "doesnt": "does not",
  "dont": "do not",
  "hadnt": "had not",
  "hasnt": "has not",
  "havent": "have not",
  "hed": "he would",
  "hell": "he will",
  "hes": "he is",
  "id": "I had",
  "ill": "I will",
  "im": "I am",
  "isnt": "is not",
  "its": "it is",
  "itll": "it will",
  "ive": "I have",
  "lets": "let us",
  "mightnt": "might not",
  "mustnt": "must not",
  "shant": "shall not",
  "shed" : "she would",
  "shell": "she will",
  "shes": "she is",
  "shouldnt": "should not",
  "thats": "that is",
  "theres": "there is",
  "theyd": "they would",
  "theyll": "they will",
  "theyre": "they are",
  "theyve": "they have",
  "wed": "we would",
  "were": "we are",
  "werent": "were not",
  "weve": "we have",
  "whatll": "what will",
  "whatre": "what are",
  "whats": "what is",
  "whatve": "what have",
  "wheres": "where is",
  "whod": "who would",
  "wholl": "who will",
  "whore": "who are",
  "whos": "who is",
  "whove": "who have",
  "wont": "will not",
  "wouldnt": "would not",
  "youd": "you would",
  "youll": "you will",
  "youre": "you are",
  "youve": "you have",
  "re":  "are",
  "wasnt": "was not",
  "well":  "will"
}


# In[31]:


# https://github.com/minerva-ml/open-solution-toxic-comments/blob/master/external_data/compiled_bad_words.txt
compiled_bad_list = pd.read_csv('https://raw.githubusercontent.com/minerva-ml/open-solution-toxic-comments/master/external_data/compiled_bad_words.txt', header=None)
compiled_bad_list = list(compiled_bad_list[0].values)


# In[32]:


# From https://github.com/kaymal/twitter-emotions/blob/master/data-preprocessing.ipynb
def preprocess_tweet(tweet):
    # To lowercase (not good for VADER)
    tweet = tweet.lower()
    
    # Remove fucking words
    for bad_word in compiled_bad_list:
        bad_candidate = ' ' + bad_word + ' '
        tweet = tweet.replace(bad_candidate, ' ')
        
    # Replace emoji unicode to text
#     tweet = emoji.demojize(tweet)
#     tweet = tweet.replace('_', ' ')
#     tweet = tweet.replace(':', ' ')
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')    
    # Remove punctuation
    tweet = tweet.replace('.', ' ')
    tweet = tweet.replace(',', ' ')
    
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    
    # Replace apostrophes to original term
    for key in apostrophes.keys():
        tweet = tweet.replace(key, apostrophes[key])
    
    #Convert @username to "user"
    tweet = re.sub('@[^\s]+', 'user', tweet)
    
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    
#     # Remove single space remaining at the front of the tweet.
#     tweet = tweet.lstrip(' ')
    
#     # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
#     tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    
#     # Convert hyperlinks ->>>> For now just replace with http
#     tweet = re.sub(r'https?:\/\/.*\/\w*', 'http', tweet)

#     #Remove @user
#     tweet = re.sub('@[^\s]+','',tweet)
    
#     # Remove tickers such as USD ($)
#     tweet = re.sub(r'\$\w*', '', tweet)
    
#     # Remove hashtags (not good for VADER)
#     tweet = re.sub(r'#\w*', '', tweet)
    
#     # Remove Punctuation and split 's, 't, 've with a space for filter
#     tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    
#     # Remove words with 2 or fewer letters
#     tweet = re.sub(r'\b\w{1,2}\b', '', tweet)

    return tweet


# In[7]:


df['text'] = df.text.apply(preprocess_tweet)
df['reply'] = df.reply.apply(preprocess_tweet)
print(df['text'][173])
print(df['reply'][3])


# In[33]:


categories_type = pd.read_json('./source/categories.json', lines=True)
categories_mapping = {v[0]: k for k, v in categories_type.to_dict('list').items()}


# ## Multilabel classification of BERT

# In[34]:


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification. This module is composed of the BERT model with a linear layer on top of the pooled output. """ 
    def __init__(self, config, num_labels=43):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
#         self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         _, pooled_output = self.roberta(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# ## Preprocessing json

# In[35]:


# https://github.com/jiesutd/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/pregenerate_training_data.py
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


# In[39]:


class Preprocess():
    def __init__(self, epochs=6, batch_size=64, max_seq_length=128, categories=None):
        self.label = categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories), output_hidden_states=False)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         self.model = BertForMultiLabelSequenceClassification.from_pretrained("roberta-base", num_labels=len(categories), output_hidden_states=False)
#         self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    
    def tokenize(self, sentance):
        return self.tokenizer.tokenize(sentance)
    
    def convert_features_to_tensors(self, corpus_text, corpus_reply, corpus_class):
        df_tokenize = []
        for sid, sentance in enumerate(corpus_text):
            sentance_reply = corpus_reply[sid]
            token_sentance = self.tokenize(sentance)
            token_reply = self.tokenize(sentance_reply)
            
            # Since max length will > max_seq_length, need to truncate pairs
            truncate_seq_pair(token_sentance, token_reply, self.max_seq_length - 3)
            
            first_sentance = ['[CLS]'] + token_sentance + ['[SEP]']
            second_sentance = token_reply + ['[SEP]']
            tokens = self.tokenizer.convert_tokens_to_ids(first_sentance + second_sentance)
            len_first = len(first_sentance)
            len_second = len(second_sentance)
            tokens_tensor = torch.tensor(tokens)
            segments_tensor = torch.tensor([0] * len_first + [1] * len_second, dtype=torch.long)
            # Convert label to one hot encoding
            label_onehot_tensor = torch.zeros([len(self.label)])
            for each_class in corpus_class[sid]:
                label_onehot_tensor[self.label[each_class]] = 1
            df_tokenize.append([tokens_tensor, segments_tensor, label_onehot_tensor])
        return df_tokenize
    
    def create_mini_batch(self, corpus):
        tokens_tensors = [sentance[0] for sentance in corpus]
        segments_tensors = [sentance[1] for sentance in corpus]
        labels_tensors = torch.stack([sentance[2] for sentance in corpus])
#         labels_tensors = labels_tensors.type(torch.LongTensor)
        # zero padding
        tokens_tensors = torch.nn.utils.rnn.pad_sequence(tokens_tensors, batch_first=True)
        segments_tensors = torch.nn.utils.rnn.pad_sequence(segments_tensors, batch_first=True)
        # attention masks to set non-zero padding position to one to let BERT focus on these tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
        return tokens_tensors, segments_tensors, masks_tensors, labels_tensors
    
    def train(self, training_data):
        # let model training on GPU
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print("device:", device)
    
        # Load pretrained model
#         model_state_dict = torch.load('./models/bert_adam_3_256')
#         self.model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(self.label), state_dict=model_state_dict)
        self.model = self.model.to(device)
        
        # training mode
        self.model.train()
        # select adam as optimizer to update weights
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        
        for epoch in tqdm(range(self.epochs), desc='Training epochs: '):
            running_loss = 0
            for data in tqdm(training_data, desc='Training progress: '):
                tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
                # Initial gradients
                optimizer.zero_grad()
                # forward pass
                loss = self.model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=labels)
                # backward
                loss.backward()
                optimizer.step()
                # Record current batch loss
                running_loss += loss.item()
            print("Epoch: {}, loss: {}".format(epoch + 1, running_loss))
        torch.save(self.model.state_dict(), './models/bert_adam_{}_{}_{}'.format(self.epochs, self.batch_size, self.max_seq_length))
        # Calculate classification accuracy
#         _, acc = self.get_predictions(model, training_data, True)
        return self.model
    
    def baseline(self, corpus_text, corpus_reply, corpus_class):
        df_tokenize = self.convert_features_to_tensors(corpus_text, corpus_reply, corpus_class)
        train_loader = torch.utils.data.DataLoader(df_tokenize, batch_size=self.batch_size, collate_fn=self.create_mini_batch)
        tuned_model = self.train(train_loader)
        return tuned_model


# In[40]:


max_seq_length = 128


# In[41]:


preprocess_module = Preprocess(epochs=3, batch_size=256, max_seq_length=max_seq_length, categories=categories_mapping)


# In[42]:


tuned_model = preprocess_module.baseline(df['text'], df['reply'], df['categories'])

