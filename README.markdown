# Named-Entity Recognition on Social Media Data

This project was created to compare multiple methods of named-entity recognition on social media data (twitter tweets). However, it is possible to also utilize the existing models on other forms of text. 

## Methods
1. Bi-directional LSTM model 
2. BERT for Token Classification 

## Paper Introduction
The goal of this project is to build a named-entity recognizer for Twitter text. In particular, we are advised to focus mainly on high accuracy and speed. The data supplied to us aggregated from Twitter tweets and has matching labels for named-entities. There are tokens B for the start of a named-entity, I for continuation of a. named-entity, and O for any other token. 

For this project, I implemented two different approaches. The first approach was a through using only PyTorch's LSTM infrastructure to create a neural network to classify the 3 different tokens. This first approach was motivated by focusing on speed, the network was fast to create. The second approach was through and implementation of HuggingFace's BertForTokenClassification. I was motivated to chose this method to focus on obtaining high accuracy. 

For both experiments, I stuck with the basic data we were supplied. I ran multiple experiments with both models. The first model I experimented with different network structures, embedding dimensions and methods, loss functions, and methods to deal with unknowns words.
For the second approach I experimented with different padding lengths, batch sizes, epochs, and loss functions. The F1 test scores for the first and second models are 0.42124 and 0.62284 respectively. 

## Future Steps:
Given more time, I would have liked to utilize differnent types of transformer models. For instance, RoBERTa-large could potentially bring in better results. 

## Special Thanks:
Thank you to Yoav Artzi for all the help with NLP tasks!

## Citations:

https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification 


https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/


https://www.aclweb.org/anthology/Q16-1026.pdf


https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede


https://gab41.lab41.org/how-to-fine-tune-bert-for-named-entity-recognition-2257b5e5ce7e
