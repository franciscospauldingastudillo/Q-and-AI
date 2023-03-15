# Q-and-AI (Q&AI)
A supervised machine learning algorithm for binary classification of human- (wikipedia) and AI- (ChatGPT) generated text


In this project, I trained an unsupervised machine learning model (Naive Bayes) affectionately called ‘Q&AI’ that classifies textual data as human- or AI-generated. Using the natural language toolkit (NLTK) library, I randomly draw 500 words from a database of English words (known as a corpus) corresponding to broadly-understood topics (e.g. airplane, Congress, apple). First, I used the Wikipedia API to download a text abstract on each topic in json format. This is our human-generated text. Second, I used the OpenAI API to ask ChatGPT to write a short summary on each topic in json format. This is our AI-generated text. The bag-of-words (BOW) model was used to transform the text into numerical arrays. In the BOW model, features of the dataset are single words or pairs of words. Feature values correspond to the number of times that a given word appears in a sample. The TF-IDF (term frequency times inverse document frequency) correction is applied to the arrays to reduce the weightage of commonly-used words (the, is) and bias intorudced by text length. Subsets of the labeled numerical data were used to train and test the Naive Bayes classifier. 

A requirements.txt file is provided to set up your virtual environment. Inside your virtual environment, run: 
pip install -r requirements.txt

