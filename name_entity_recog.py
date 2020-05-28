import string
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import BertTokenizer, BertForTokenClassification, BertConfig

from sklearn.metrics import confusion_matrix


def load_data(data_path):
    with open(data_path) as data:
        sentences = []
        tags = []
        curr_sen = []
        curr_tag = []
        i = 0
        for phrase in data:
            if phrase != '\n':
                word, tag = phrase.split('\t')
                curr_sen.append(word)
                # strip removes the trailing '\n'
                curr_tag.append(tag.strip())
            if phrase == '\n':
                sentences.append(curr_sen)
                tags.append(curr_tag)
                curr_sen = []
                curr_tag = []
                
    return sentences, tags
def load_test_data(data_path):
    with open(data_path) as data:
        sentences = []
        test_data = []
        for phrase in data.read().strip().split('\n\n'):
            phrase = phrase.strip()
            lines = phrase.split('\n')
            test_data.append([line.strip() for line in lines])
    return test_data

def create_vocab_dicts(corpus):
    vocab_counts = {}
    vocab_ids = {}
    id_num = 1
    vocab_counts['-UNK-'] = 0
    vocab_ids['-UNK-'] = 0
    
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            curr_word = corpus[i][j]
            
            # make dict with key = vocab, value = id
            if curr_word not in vocab_ids:
                vocab_ids[curr_word] = id_num
                id_num += 1
                
            # make dict with key = vocab, value = freq
            if curr_word not in vocab_counts:
                vocab_counts[curr_word] = 1
            else:
                vocab_counts[curr_word] += 1
                
    return vocab_counts, vocab_ids
def preprocess(corpus, vocab_count_dict, word_threshold):
    
    num_hashtags = 0
    num_mentions = 0
    num_puncs = 0
    num_links = 0
    num_rare = 0
    num_digit = 0
    special_tags = ['-HASHTAG-', '-MENTION-', '-PUNC-', '-LINK-']
    
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            curr_word = corpus[i][j]
        
            if curr_word[0] == '#':
                corpus[i][j] = '-HASHTAG-'
                num_hashtags += 1
                
            if curr_word[0] == '@':
                corpus[i][j] = '-MENTION-'
                num_mentions += 1
            
            if curr_word[0] in '!$%&\'()*+,-./:;<=>?[\\]^_`{|}~':
                corpus[i][j] = '-PUNC-'
                num_puncs += 1
            
            if (curr_word[0:4] == 'http') or (curr_word[0:3] == 'www'):
                corpus[i][j] = '-LINK-'
                num_links += 1
    
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            curr_word = corpus[i][j]
            
            if curr_word in special_tags:
                continue
                
            if curr_word.isdigit():
                corpus[i][j] = '-DIGIT-'
                num_digit += 1
                
            if vocab_count_dict[curr_word] <= word_threshold:
                corpus[i][j] = '-RARE-'
                num_rare += 1
                    
    vocab_counts, vocab_ids = create_vocab_dicts(corpus)

    print('Total Changed Words')
    print('-------------------')
    print('Hashtags:      {}'.format(num_hashtags))
    print('Mentions:      {}'.format(num_mentions))
    print('Punctuation:   {}'.format(num_puncs))
    print('Links:         {}'.format(num_links))
    print('Rare:          {}'.format(num_rare))
    print('Digit:         {}'.format(num_digit))
    print('-------------------')
    print('Vocab Size:    {}'.format(len(vocab_counts)))
    
    return corpus, vocab_counts, vocab_ids
def words_to_ids(corpus, vocab2ID_dict):
    
    num_unk = 0
    
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            try:
                corpus[i][j] = vocab_ids[corpus[i][j]]
            except:
                corpus[i][j] = vocab_ids['-UNK-']
                num_unk += 1
    
    print('Number of Unknown Words: {}'.format(num_unk))
    
    return corpus
def tags_to_vectors(tags):
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            curr_tag = tags[i][j]
            if curr_tag == 'B':
                tags[i][j] = [1, 0, 0]
            elif curr_tag == 'I':
                tags[i][j] = [0, 1, 0]
            elif curr_tag == 'O':
                tags[i][j] = [0, 0, 1]
            else:
                print('bro what')
                break
    return tags
def tags_to_onehot(tags):
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            curr_tag = tags[i][j]
            if curr_tag == 'B':
                tags[i][j] = 2
            elif curr_tag == 'I':
                tags[i][j] = 1
            elif curr_tag == 'O':
                tags[i][j] = 0
            else:
                print('Indevid tag!')
                break
    return tags

class Net(nn.Module):
    
    def __init__(self, num_epochs, learning_rate, momentum, 
                 vocab_size, embedding_dim, lstm_dim1, lstm_dim2, lstm_dim3):
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.num_epochs = num_epochs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim1 = lstm_dim1
        self.lstm_hidden_dim2 = lstm_dim2
        self.lstm_hidden_dim3 = lstm_dim3
        
        super(Net, self).__init__()
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.lstm1 = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim1)
        self.lstm2 = nn.LSTM(self.lstm_hidden_dim1, self.lstm_hidden_dim2)
        self.lstm3 = nn.LSTM(self.lstm_hidden_dim2, self.lstm_hidden_dim3)
        
        self.fc1 = nn.Linear(self.lstm_hidden_dim3, int(self.lstm_hidden_dim3/2))
        self.fc2 = nn.Linear(int(self.lstm_hidden_dim2/2), 3)
        
    def forward(self, x):
        
        x = self.embedding(x)

        x = x.view(1, -1, x.shape[1])
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return F.log_softmax(x, dim = 1)
        #return x
    
    def train_and_evaluate(self, data, labels, dev_data, dev_labels):

        start_time = time.time()

        total_training_loss = []
        total_dev_loss = []
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr = self.learning_rate, momentum = self.momentum)

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            print('Starting Epoch: {}'.format(epoch))
            print('------------------------')
            train_loss = 0

            for i, (sentence, label) in enumerate(zip(data, labels)):            
                sentence, label = torch.LongTensor(sentence), torch.LongTensor(label)

                optimizer.zero_grad()

                outputs = net(sentence)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()    
                train_loss += loss.item()

            total_training_loss.append(train_loss)

            with torch.no_grad():
                running_dev_loss = 0
                for i, (sentence_val, label_val) in enumerate(zip(dev_data, dev_labels)):
                    sentence_val, label_val = torch.LongTensor(sentence_val), torch.LongTensor(label_val)

                    net.eval()

                    prediction = net(sentence_val)

                    dev_loss = criterion(prediction, label_val)
                    running_dev_loss += dev_loss.item()

            total_dev_loss.append(dev_loss)

            epoch_end = time.time()
            total_epoch_time = round((epoch_end - epoch_start) / 60, 2)
 
            print('Epoch: {}\nTraining Loss: {}\nVal Loss: {}'.format(epoch, train_loss, dev_loss))
            print('Epoch run time: {} minutes\n'.format(total_epoch_time))

        end_time = time.time()
        total_time = round((end_time - start_time) / 60, 2)

        print('Finished training in {} minutes'.format(total_time))
        return total_training_loss, total_dev_loss
def predict_and_transform(network, data, tag_index_dict):
    predictions = []
    
    with torch.no_grad():
        for sample in data:
            sample = torch.LongTensor(sample)
            predicted_label = network(sample)
            predictions.append(predicted_label)

    trans_pred = []
    for prediction in predictions:
        curr_sample = []
        for label in prediction:
            label = np.argsort(-label.numpy())[0]
            trans_label = tag_index_dict[label]
            curr_sample.append(trans_label)
        trans_pred.append(curr_sample)
    
    return trans_pred

def export_predictions(file_path, predictions):
    with open(file_path, 'w') as f:
        for sample in predictions:
            for label in sample:
                line = label + '\n'
                f.write(line)
            f.write('\n')

def transform_confusion_data(data):
    token_map = {'B': 2, 'I': 1, 'O': 0}

    for i in range(len(data)):
        data[i] = token_map[data[i]]

    return data

def create_confusion_matrix(dev_true, dev_pred):
    dev_true = transform_confusion_data(dev_true)
    dev_pred = transform_confusion_data(dev_pred)

    conf_mat = confusion_matrix(dev_true, dev_pred)

    ax = sns.heatmap(conf_mat, annot = True, fmt = 'g', cmap = 'Blues')
    ax.set_title('Confusion Matrix')

    pass

### start second model

def train_and_evaluate(bert_tokenizer, tag_index_dict, model, num_epochs, train_loader, optimizer, device, dev_loader, index_tag_dict):
    total_training_loss = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i, batch in enumerate (train_loader):
            batch = tuple(sample.to(device) for sample in batch)

            inputs, labels, mask = batch[0], batch[1], batch[2]
    
            outputs = model(inputs, token_type_ids = None, attention_mask = mask, labels = labels)

            loss, probs = outputs[:2]
            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 1)

            optimizer.step()
            model.zero_grad()

        loss = train_loss/len(train_loader)
        total_training_loss.append(loss)

        model.eval()
        dev_loss = 0
        dev_preds =  []

        for i,batch in enumerate(dev_loader):
            batch = tuple(sample.to(device) for sample in batch)
            
            dev_inputs, dev_labels, dev_mask = batch[0], batch[1], batch[2]

        with torch.no_grad():
            outputs = model(dev_inputs,token_type_ids = None, attention_mask = dev_mask, labels = dev_labels)

            loss, probs = outputs[:2]
            dev_loss += loss.item()

            preds_mask = (dev_labels != -100)
            probs = probs.detach().cpu().numpy()
            filtered_probs = probs[preds_mask.cpu().squeeze()]
      
            preds = np.argmax(filtered_probs, axis = 1)
            label_ids = torch.masked_select(dev_labels,(preds_mask == 1))
            label_ids = label_ids.to('cpu').numpy()

        labels = [index_tag_dict[label_id] for label_id in label_ids]
        predictions = [index_tag_dict[pred] for pred in preds]

        dev_preds.extend(predictions)

        print('Epoch: {}, Loss: {} '.format(epoch+1, loss))

    return model, dev_preds



def transform_bert_data(sentences, labels, tokenizer, tag_to_idx):
    max_length = 140

    # tokenize text
    tokenized_texts = []
    for sentence, label in zip(sentences, labels):
        tokenized_texts.append(tokenize_sequence(sentence, label, tokenizer))

    # create tokens and mask
    tokens, mask = [], []
    for text in tokenized_texts:
        tokens.append(text[0])
        mask.append(text[1])

    # add the padded data and attention masks
    padded_data, attn_masks = [], []
    for tokens in tokens:
        tokens_tmp = ['[CLS]'] + tokens
        padded_tokens = tokens_tmp + ['[PAD]' for _ in range(max_length - len(tokens_tmp))]
        padded_tokens = tokenizer.convert_tokens_to_ids(padded_tokens)
        padded_data.append(padded_tokens)
    attn_masks = [[float(word > 0) for word in sequence] for sequence in padded_data]

    # add the padded labels
    padded_labels = []
    for labels in labels:
        padded_labels = [tag_to_idx.get(l) for l in labels]
        padded_labels = [-100] + padded_labels + [-100]
        padded_labels = padded_labels + [-100 for _ in range(max_length - len(padded_labels))]
        padded_labels.append(padded_labels)

    for words, tags in zip(padded_data, padded_labels):
            if words[-1] == tokenizer.vocab['[PAD]']:
                continue
            else:
                words[-1] = tokenizer.vocab['[SEP]']
                tags[-1] = -100

    return tokens, labels, padded_data, attn_masks, padded_labels  

def transform_bert_test_data(sentences, labels, tokenizer, tag_to_idx):
    max_length = 140

    # tokenize text
    tokenized_texts = []
    for sentence in sentences:
        tokenized_texts.append(tokenize_sequence_test(sentence, tokenizer))

    # create tokens and mask
    tokens, mask = [], []
    for text in tokenized_texts:
        tokens.append(text[0])
        mask.append(text[1])

    # add the padded data and attention masks
    padded_data, attn_masks = [], []
    for tokens, mask in zip(tokens, mask):
        tokens_temp = ['[CLS]'] + tokens
        padded_tokens = tokens_temp + ['[PAD]' for _ in range(max_length - len(tokens))]
        padded_tokens = tokenizer.convert_tokens_to_ids(padded_tokens)
        padded_data.append(padded_tokens)
    attn_masks = [[float(word > 0) for word in sequence] for sequence in padded_data]

    # finalize padding
    for words in padded_data:
        if words[-1] == tokenizer.vocab['[PAD]']:
            continue
        else:
            words[-1] = tokenizer.vocab['[SEP]']

    return tokens, mask, padded_data, attn_masks,

def tokenize_sequence(sentence, label, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, labels):
        tokens = tokenizer.tokenize(word)
        num_tokens = len(tokens)
        tokenized_sentence.extend(tokens)

    if num_tokens > 1:
        labels.extend([label] + ['[PAD]',] * (num_tokens - 2) + ['[PAD]'])
    if num_tokens == 1:
        labels.extend([label])

    return tokenized_sentence, labels

# helper function
def tokenize_sequence_test(sentence, tokenizer):
    tokenized_sentence, mask = [], []

    for word in sentence:
        tokens = tokenizer.tokenize(word)
        num_tokens = len(tokens)
        tokenized_sentence.extend(tokens)

    if num_tokens == 1:
        mask.extend([1])
    if num_tokens > 1:
        mask.extend([1] + [0] * (num_tokens - 2) + [0])

    return tokenized_sentence, mask

def bert_data_parser(corpus, vocab_counts, vocab_ids, tags, 
                    dev_corpus, dev_vocab_counts, dev_vocab_ids, dev_tags,
                    test_corpus, test_vocab_counts, test_vocab_ids,
                    tokenizer, tag_index_dict, batch_size
                    ):

    tokens, labels, padded_data, attn_masks, padded_labels = transform_bert_data(corpus, tags, tokenizer, tag_index_dict)
    dev_tokens, dev_labels, dev_padded_data, dev_attn_masks, dev_padded_labels = transform_bert_data(dev_corpus, dev_tags, tokenizer, tag_index_dict)
    test_tokens, test_mask, test_padded_data, test_attn_masks = transform_bert_test_data(test_corpus, tokenizer, tag_index_dict)
    
    train_data = TensorDataset(torch.tensor(padded_data), torch.tensor(padded_labels), torch.tensor(attn_masks))
    dev_data = TensorDataset(torch.tensor(dev_padded_data), torch.tensor(dev_padded_labels), torch.tensor(dev_attn_masks))
    test_data = TensorDataset(torch.tensor(test_padded_data), torch.tensor(test_attn_masks))

    train_rand = RandomSampler(train_data)
    dev_seq = SequentialSampler(dev_data)
    test_seq = SequentialSampler(test_data)

    train_loader = DataLoader(train_data, batch_size, sampler = train_rand)
    dev_loader = DataLoader(dev_data, batch_size, sampler = dev_seq)
    test_loader = DataLoader(test_data, batch_size, sampler = test_seq)

    return train_loader, dev_loader, test_loader


if __name__=='__main__':
    # decide which model to use
    model = 1

    # load train data
    train_data = 'data/train/train.txt'
    sentences, tags = load_data(train_data)
    vocab_counts, vocab_ids = create_vocab_dicts(sentences)
    corpus, vocab_counts, vocab_ids = preprocess(sentences, vocab_counts, 1)
    corpus = words_to_ids(corpus, vocab_ids)
    tags = tags_to_onehot(tags)

    # load dev data
    dev_path = 'data/dev/dev.txt'
    dev_data, dev_tags = load_data(dev_path)

    dev_vocab_counts, dev_vocab_ids = create_vocab_dicts(dev_data)
    dev_corpus, dev_vocab_counts, dev_vocab_ids = preprocess(dev_data, dev_vocab_counts, 1)

    dev_corpus = words_to_ids(dev_corpus, vocab_ids)

    dev_tags = tags_to_onehot(dev_tags)

    # load test data
    test_path = 'data/test/test.nolabels.txt'
    test_data = load_test_data(test_path)

    test_vocab_counts, test_vocab_ids = create_vocab_dicts(test_data)
    test_corpus, test_vocab_counts, test_vocab_ids = preprocess(test_data, test_vocab_counts, 1)

    test_corpus = words_to_ids(test_corpus, vocab_ids)
    if model == '1':
        # params for model 1:
        num_epochs = 1
        learning_rate = 0.01
        momentum = 0.9
        vocab_size = len(vocab_ids)
        embedding_dim = 200
        lstm_dim1 = 150
        lstm_dim2 = 125
        lstm_dim3 = 100

        net = Net(num_epochs, learning_rate, momentum, vocab_size, embedding_dim, lstm_dim1, lstm_dim2, lstm_dim3)
        total_train_loss, total_dev_loss = net.train_and_evaluate(corpus, tags, dev_corpus, dev_tags)   
    else:
        # params for model 2:
        num_epochs = 1
        learning_rate = 0.001
        momentum = 0.9
        batch_size = 20
        num_epochs = 7

        # start set up
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # tokenize data to match BERT requirements 
        index_tag_dict = {'B': 0, 'I': 1, 'O': 2, '[PAD]': -100}
        tag_index_dict = {0: 'B', 1: 'I', 2: 'O', -100: '[PAD]'}
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        train_loader, dev_loader, test_loader = bert_data_parser(corpus, vocab_counts, vocab_ids, tags, 
                                                                                    dev_corpus, dev_vocab_counts, dev_vocab_ids, dev_tags,
                                                                                    test_corpus, test_vocab_counts, test_vocab_ids,
                                                                                    bert_tokenizer, tag_index_dict, batch_size
                                                                                    )

        bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels = 3)
        bert.to(device)

        optimizer = optim.SGD(bert.parameters(), lr = learning_rate, momentum = momentum)
        model, dev_pred = train_and_evaluate(bert_tokenizer, tag_index_dict, model, num_epochs, train_loader, optimizer, device, dev_loader, index_tag_dict)

        test_pred = predict_and_transform(bert, test_data, tag_index_dict)


    # creating results for model 1!
    tag_index_dict = {0: 'B', 1: 'I', 2: 'O'}
    dev_pred = predict_and_transform(net, dev_data, tag_index_dict)

    pred_file_path = 'results/dev/dev_pred.out'
    export_predictions(pred_file_path, dev_pred)

    create_confusion_matrix(dev_corpus, dev_pred)