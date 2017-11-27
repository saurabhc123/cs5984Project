import gensim
import numpy as np
import string


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords




pretrained_word_vector_binary = 'c:\\temp\\GoogleNews-vectors-negative300.bin'






class word2vec:
    model = None
    models = {}
    stop = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    def get_model(self):
        if not self.model:
            print("Loading word2vec model from file:{}".format(pretrained_word_vector_binary))
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word_vector_binary, binary=True)
            model.init_sims(replace=True)  # we are no longer training the model so allow it to trim memory
            word2vec.model = model

        return self.model


    def get_model_from_file(self,name):
        if not name in word2vec.models:
            sentences = []
            with open(name,"r") as f:
                sentences = map(lambda x: word2vec.extract_sentence(x).split(), f.readlines())
            file_model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1, window=5)
            # sentences = []
            # sentences.append('ebola threat real allow african conference nyc risky stupid wrong'.split())
            # file_model.similar_by_vector(sentences)
            word2vec.models[name] = file_model.wv
        return word2vec.models[name]


    def extract_sentence(self, line):
        lineContent = line.split(',')
        return (lineContent[1] if len(lineContent) > 1 else lineContent[0])

    def avg_feature_vector(self, words, model, num_features , index2word_set):
        # function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        model = self.get_model()

        # list containing names of words in the vocabulary
        # index2word_set = set(model.index2word) this is moved as input param for performance reasons
        for word in words:
            if word in index2word_set:
                nwords += 1
                featureVec = np.add(featureVec, model[word])

        if (nwords > 0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def get_word_vector(self, word):
        model = self.get_model()
        return model[word]

    def get_sentence_vector(self, sentence):
        featureVec = np.zeros((1,300))
        nwords = 0
        clean_sentence = self.clean_sent(self.wordnet_lemmatizer, sentence)
        model = self.get_model()
        words = clean_sentence.rstrip().split(" ")
        if(len(words) == 0):
            return featureVec
        for word in words:
            try:
                word_vector = model[word]
                nwords += 1
                #print word_vector
                featureVec = np.add(featureVec, word_vector)
            except:
                pass #Swallow exception
        if (nwords > 0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def read_from_file(self, name, model):
        with open(name,"r") as f:
            lines = f.readlines()
            tweets_only = map(lambda line: line.split(',')[1].rstrip().split(" "), lines)
            lables_only = map(lambda line: line.split(',')[0].rstrip(), lines)
            vecs = map(lambda t: self.avg_feature_vector(t, model, 300, model.index2word),tweets_only)
            return zip(lables_only,vecs)

    def full_pipeline(self, lem, word):
        word = word.lower()
        word = word.translate(string.punctuation)
        for val in ['a', 'v', 'n']:
            word = lem.lemmatize(word, pos=val)
        return word


    def clean_sent(self, lem, sent):
        #sent = unicode(sent,errors='ignore')
        words = sent.replace(","," ").replace(";", " ").replace("#"," ").replace(":", " ").replace("@", " ").split()
        filtered_words = filter(lambda word: word.isalpha() and len(word) > 1 and word != "http" and word != "rt", [self.full_pipeline(lem, word) for word in words])
        return ' '.join(self.filter_stopwords(filtered_words))

    def filter_stopwords(self, words):
        return filter(lambda word: word not in self.stop, words)





if __name__ == "__main__":
    sentence = "Gala Bingo clubs bought for 241m: The UK's largest High Street bingo operator, Gala, is being taken over by_ https://t.co/HzeeykJUd3"
    wv = word2vec()
    clean_sentence = wv.clean_sent(wv.wordnet_lemmatizer,sentence)
    print(clean_sentence)

    model = word2vec().get_model()
    # you can find the terms that are similar to a list of words and different from
    # another list of words like so
    print(model.most_similar(positive=['hurricane'], negative=['isaac']))

    # you can also get the vector for a specific word by doing
    print(model['hurricane'])

    # you can ask for similarity by doing
    print(model.similarity('hurricane', 'shooting'))

    print("done")