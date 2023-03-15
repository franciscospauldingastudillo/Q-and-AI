# Q-and-AI (Q&AI)
tldr: a supervised machine learning algorithm for binary classification of human- (wikipedia) and AI- (ChatGPT) generated text

In this project, I trained an unsupervised machine learning model (Naive Bayes) affectionately named ‘Q&AI’ that classifies textual data as human- or AI-generated. Using the natural language toolkit (NLTK) library, I randomly draw 'n' words from a database of English words (known as a corpus) corresponding to broadly-understood topics (e.g. airplane, Congress, apple). Next, I use the Wikipedia API to download a text abstract on each topic in json format. This is our human-generated text. Then, I use the OpenAI API to ask ChatGPT to write a short summary on each topic in json format. This is our AI-generated text. The bag-of-words (BOW) model was chosen to transform the text into numerical arrays. In the BOW model, features of the dataset are single words or pairs of words. The numerical value assigned to each feature corresponds to the number of times that the respective word appears in a sample. The TF-IDF (term frequency times inverse document frequency) correction is applied to the arrays to reduce bias from commonly-used words (the, is) and text length. Subsets of the labeled numerical data were used to train and test the Naive Bayes classifier. The previous steps were combined into a pipeline. Optimal model performance was obtained through a grid search of several tuning parameters. The tuned model was capable of detecting AI-generated text with an accuracy of 75%.

A requirements.txt file is provided to set up your virtual environment. Inside your virtual environment, run: 
"pip install -r requirements.txt"

Before running the model,
1) copy and paste your OpenAI user key into gpt_query(). You can obtain this key from your OpenAI account.
2) change the path to your working directory.

To run the model:
"python3 ./qai.py nwords"
where nwords is the number of topics that you want the model to include in your dataset. 

The data collection step can take minutes to hours, depending on nwords. For those that wish to run the model and skip the data collection step,
I have provided a json file with 598 words: data-598.json. To run the model with the pre-packaged data, run:
"python3 ./qai.py 598"

Have fun!
