# TLDR: Q&AI is a binary text-classification algorithm. It uses a Naive Bayes model to differentiate
# ai- (chatgpt) and human-generated (wikipedia) text. If developed further, the model could be used 
# to verify multiple text sources or detect AI plagiarism. 

# import libraries
import sklearn      # scikit-learn (for machine learning and data manip.)
import openai       # chatgpt api (application programming interface)
import json         #
import numpy as np
import time        
import os           # command line interface with python
import nltk         # natural-language toolkit
import requests     # for http reqeusts
import pandas       # for dataframe

# To submit a query through the openai api, you must provide a key
# associated your openai account, an upper limit on the number of words in 
# the response (max_tokens), and a temperature (randomness in the response).
# The api returns a json object, from which the reponse is extracted as text.
def gpt_query(word):
    # user note: ChatGPT does not return reponses with the exact # of words requested
    openai.api_key = "your-unique-key"
    model_engine = "text-davinci-002"
    prompt = "Write an extended summary about this noun: '%s'. Use exactly 500 words in your response."%(word)
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=1,
    )
    message = completions.choices[0].text
    message = message.replace('\n','')
    return message

# In this query request through the wikipedia api, the user provides a subject and 
# signals that only the abstract is to be extracted. If a page on the 
# queried subject exists,  the requested data is returned as a json object, 
# which must be parsed  to obtain the abstract text. If no page exists, 
# an empty string is returned. If multiple pages are found (disambiguation), 
# the first page is selected and a new search is carried out recursively on this page. 
# The abstract length is capped at nwords, an operation performed by the helper function, wik_crop().

def wik_query(subject,nwords): 
    ##########################################################
    def wikapi(params):
        url = 'https://en.wikipedia.org/w/api.php'
        response = requests.get(url, params=params)
        data = response.json()
        return data
    ##########################################################
    def pick_title(subject):
        params = {
                'action': 'query',
                'format': 'json', 
                'titles': subject,
                'prop': 'links',
                }
        data = wikapi(params)
        page = next(iter(data['query']['pages'].values()))
        links = next(iter(page['links']))
        title = links['title']
        return title
    ##############################################################
    def wik_crop(text,nwords):
        ##########################################################
        maxchar  = min(10*nwords,len(text)-1) # cap on char. count
        spaces   = 0                          # proxy for word counter
        newtext = ''                          # abbreviated text (output)
        it      = -1                          # iteration #
        ##########################################################
        if (text.count(' ')+1)<nwords:
            print("the abstract has fewer words than nwords")
            newtext = text.replace('\n','')
            return newtext
        else:
            while (spaces+1) < nwords: 
                it+=1
                #################
                if it>maxchar:
                    print('while loop exceeded maximum iterations')
                    return '' 
                #################
                char = text[it]
                if char==' ':
                    spaces += 1
                #################
                newtext = newtext + char
                #################
            newtext=newtext[0:-1]+'.'
            newtext = newtext.replace('\n','')
            return newtext
    ##########################################################
    nchar = 15*nwords # of characters to extract from the abstract (rough guess)
    # Step 1: Submit a query request through the Wikipedia API
    params = {
            'action': 'query',
            'format': 'json', 
            'titles': subject,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            }
    data = wikapi(params)
    page = next(iter(data['query']['pages'].values()))
    # Scenario 1: the page does not exist
    if 'missing' in page:
        print("page on %s not found. returning empty string\n"%(subject))
        return ''
    # Scenario 2: disambiguation error (multiple pages on the same subject)
    elif 'may refer to' in page['extract']:
        title = pick_title(subject)
        print("found multiple pages on %s. picking %s\n"%(subject,title))
        params = {
            'action': 'query',
            'format': 'json', 
            'titles': title,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            }
        data = wikapi(params)
        page = next(iter(data['query']['pages'].values()))
        summary = page['extract'][:nchar]
        summary = wik_crop(summary,nwords)
        return summary
    else:
    # Scenario 3: page was found and extract the first n characters from the abstract
        print("found page on %s\n"%(subject))
        summary = page['extract'][:nchar]
        summary = wik_crop(summary,nwords)
        return summary

# To obtain a list of searchable subjects of length nnouns, the user downloads and then loads a corpus
# (i.e., database of English words) and only extracts words that are labeled as 'natural nouns'.
# Repeat words are omitted.
def get_nouns(nnouns):
    nltk.download("brown") 
    nltk.download('averaged_perceptron_tagger')
    from nltk.corpus import brown
    nouns = []
    for word,pos in nltk.pos_tag(brown.words()):
        if pos.startswith("NN"):
            if word not in nouns: # no repeats
                nouns.append(word)
            if len(nouns)==nnouns:
                break    
    return nouns    

# The data collection process can be time-intensive. To save time, 
# the query subject and the text reponses are saved in json format.
# The user can then load the json data without repeating the 
# data collection step unnecessarily.
def save(gpt,wik,nouns):
    # Save the lists to a json object
    with open("data-%d.json"%(len(nouns)), "w") as f:
        json.dump({"nouns":nouns,"gpt": gpt, "wik": wik}, f)
def load(nnouns):
    # Load the lists from a json object
    with open("data-%s.json"%(nnouns), "r") as f:
        data  = json.load(f)
        nouns = data["nouns"]
        gpt   = data["gpt"]
        wik   = data["wik"]
    return {'nouns':nouns,'gptdef':gpt,'wikdef':wik}

##############################################################################
##############################################################################
nnouns = sys.argv[1] # the number of queried words is a *command-line argument*
nouns  = []          # list of nouns
wikdef = []          # list of wikipedia responses
gptdef = []          # list of chatgpt responses
pos = 0              # current index in list of nnouns
usepipeline = True   # combine machine learning steps in a 'pipeline'
##############################################################################
fname = 'data-%d.json'%(nnouns)
if not os.path.exists(fname): # generate and save a dataset
    nouns = get_nouns(nnouns)
    fnouns = []
    for n in range(len(nouns)):
        if not n%10:
            print('%d of %d nouns'%(n,nnouns))
        noun = nouns[n]
        time.sleep(2)
        gptdef.append(gpt_query(noun))
        wikdef.append(wik_query(noun,int(gptdef[pos].count(' '))+1))
        # if wiki lookup fails, remove  key-value pairs where value=''
        if (wikdef[-1]=='' and len(wikdef)):
            gptdef = gptdef[0:-1]
            wikdef = wikdef[0:-1]
        else:
            fnouns.append(noun)
            pos+=1 # index of the next searchable noun
        # save a file at every 100 nouns
        if not len(fnouns)%1e2:
            save(gptdef,wikdef,fnouns)
    nouns = fnouns # overwrite the original list
    save(gptdef,wikdef,nouns)
else: # load dataset
    data = load(nnouns)
    gptdef = data['gptdef']
    wikdef = data['wikdef']
    nouns  = data['nouns']
##############################################################################
# Combine the lists of nouns (cnouns), AI- and human-generated responses (corpus),
# and binary targets (0=chatgpt,1=wikipedia).
cnouns = nouns+nouns
corpus = gptdef+wikdef
target = np.ones([len(corpus)],dtype='int')
target[0:len(gptdef)] = 0
##############################################################################
# To leverage machine learning, we must convert the text responses 
# into numerical arrays, where each column is a feature. 
# To vectorize the data, I use the bag-of-words model. 
# In this model, each *unique* word in the dataset is a feature. 
# The number of possible features is equal to the set of 
# all unique words in the AI and human-generated responses. 
# In a sample, the value of a specific feature (i.e. word)
# is the number of times that the word is used in the response.
##############################################################################
# Credit is due to this great article on text classification:
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
##############################################################################
if not usepipeline:
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    # fit_transform: learns the vocabulary dictionary and returns document-term matrix
    X  = vectorizer.fit_transform(corpus)
    # get_feature_names(): returns the list of features (unique words)
    #print(vectorizer.get_feature_names())
    # This shows each row of the vectorized corpus
    #print(X.toarray())

    # Explanation of the TF-IDF correction:
    # The numerical vectors are biased towards responses with more words.
    # The Term Frequency (TF) correction normalizes the vectors by the 
    # total # of words in each definition. The inverse document frequency (IDF)
    # correction reduces the weightage of commonly used words in all responses
    # such as (the, is, an, etc.).

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)

    # package the data into tuples
    df = pd.DataFrame(list(zip(cnouns,X_tfidf,target)),columns=['key','value','target'])

    # create a training/test split with the numerical data
    X = df['value']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.75)

    # train a Naive Bayes model with the training set
    from sklearn.naive_bayes import MultinomialNB
    text_clf = MultinomialNB().fit(X_train, y_train)
else: 
    # Build a pipeline to perform all the above operations
    # 'vect', 'tfidf', and 'clf' are arbitrary.
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                        ])
    text_clf = text_clf.fit(X_train, y_train)

# test the performance of Naive Bayes on the test set
predicted = text_clf.predict(X_test)
print('untuned model is %.2f%% accurate'%(np.mean(predicted==y_test)))

# use grid search to get predictions with different tuning parameters
# and optimize performance.
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range':[(1,1),(1,2)], # unigram vs bigram vectorization
              'tfidf__use_idf':(True,False),     # reduce weighting for common words
              'clf__alpha':(1e-2,1e-3),          # smoothing parameter
             }
              
gs_clf = GridSearchCV(text_clf,parameters,n_jobs=-1)
gs_clf = gs_clf.fit(X_train,y_train)
# to see the best mean score and the parameters
print('tuned model is %.2f%% accurate'%(gs_clf.best_score_))
print('the optimal parameters are '%(gs_clf.best_params_))