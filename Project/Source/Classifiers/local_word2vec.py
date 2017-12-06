import gensim
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import Placeholders


def avg_feature_vector(words, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # list containing names of words in the vocabulary
    # index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def read_from_file(name,model):
    with open(name,"r") as f:
        lines = f.readlines()
        tweets_only = map(lambda line: line.split(',')[1].rstrip().split(" "), lines)
        lables_only = map(lambda line: line.split(',')[0].rstrip(), lines)
        vecs = map(lambda t: avg_feature_vector(t, model, 300, model.index2word),tweets_only)
        return zip(lables_only,vecs)

def write_to_file(lv,file):
    with open(file,"w") as f:
        for l,v in lv:
            f.write(str(l) + "," + ','.join((str(x)for x in np.nditer(v))) + "\n")

class local_word2vec:
    model = None
    models = {}
    stop = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()


    def get_model(self):
        if not local_word2vec.model:
            sentences = wv.read_csv("Datasets/Kaggle/Kaggle-str-process.csv")
            model = wv.get_model_from_sentences(sentences)

        return model


    def get_model_from_file(self, name):
        if not name in local_word2vec.models:
            sentences = []
            with open(name,"r") as f:
                sentences = map(lambda x: local_word2vec.extract_sentence(x).split(), f.readlines())
            file_model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1, window=5)
            local_word2vec.models[name] = file_model.wv
        return local_word2vec.models[name]

    def get_sentence_vector(self, sentence):
        featureVec = np.zeros((1, Placeholders.n_inputs))
        nwords = 0
        clean_sentence = self.clean_sent(sentence)
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

    def get_sentence_matrix(self, sentence):
        sentence_matrix = []
        featureVec = np.zeros((1,Placeholders.n_inputs))
        nwords = 0
        clean_sentence = self.clean_sent(sentence)
        model = self.get_model()
        words = clean_sentence.rstrip().split(" ")
        if(len(words) == 0):
            return sentence_matrix
        for word in words:
            try:
                word_vector = model[word]
                nwords += 1
                #print word_vector
                sentence_matrix.append(word_vector)
            except:
                pass #Swallow exception
        padded_sentence_matrix = self.getPaddedSentenceMatrix(np.array(sentence_matrix))
        return padded_sentence_matrix

    def clean_sent(self, sent):
        #sent = unicode(sent,errors='ignore')
        words = sent.replace(","," ").replace(";", " ").replace("#"," ").replace(":", " ").replace("@", " ").split()
        words = filter(lambda word: word.isalpha() and len(word) > 1 and word != "http" and word != "rt", [self.full_pipeline(self.wordnet_lemmatizer, word) for word in words])
        return ' '.join(self.filter_stopwords(words))

    def filter_stopwords(self, words):
        return filter(lambda word: word not in self.stop, words)

    def get_sentence_vector_ex(self, sentence):
        try:
            sentence_matrix = self.get_sentence_matrix(sentence)
            return sentence_matrix.reshape((1, Placeholders.n_steps * Placeholders.n_inputs))
        except:
            print("Error with sentence:" + sentence)
        return np.zeros((1, Placeholders.n_steps* Placeholders.n_inputs))

    def getPaddedSentenceMatrix(self, sentenceMatrix):
        wordCount = Placeholders.n_steps
        #print(sentenceMatrix.shape)
        return np.vstack((sentenceMatrix,
                        np.zeros((wordCount - np.shape(sentenceMatrix)[0], np.shape(sentenceMatrix)[1]),
                        dtype=np.float32)))


    def get_model_from_sentences(self, sentences):
            file_model = gensim.models.Word2Vec(sentences=sentences, size=50, min_count=1, window=5)
            return file_model.wv

    def full_pipeline(self, lem, word):
        word = word.lower()
        word = word.translate(string.punctuation)
        for val in ['a', 'v', 'n']:
            word = lem.lemmatize(word, pos=val)
        return word

    def extract_sentence(self, line):
        lineContent = line.split(',')
        return (lineContent[1] if len(lineContent) > 1 else lineContent[0])

    def clean_string(self, s):
        tokens = word_tokenize(s)
        tokens = [w.lower() for w in tokens]

        stripped = [w.translate(None, string.punctuation) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        words = list(map(self.wordnet_lemmatizer.lemmatize, words))
        return words

    def read_csv(self, filename):
        sentences = []
        idx = 0
        with open(filename) as f:
            for line in f:
                if idx == 0:
                    idx = 1
                    continue
                cells = line.split(',')
                if len(cells) > 8:
                    desc = cells[2]
                    sentences.append(self.clean_string(desc))
                    body = cells[8]
                    sentences.append(self.clean_string(body))
            t = []
            for sentence in sentences:
                if len(sentence) != 0:
                    t.append(sentence)
            return t






if __name__ == "__main__":
    wv = local_word2vec()
    model = wv.get_model()
    vec = model['man']
    print(vec.shape)
    sentence = "Gala Bingo clubs bought for 241m: The UK's largest High Street bingo operator, Gala, is being taken over by_ https://t.co/HzeeykJUd3"
    clean_sentence = wv.clean_sent(sentence)
    print(clean_sentence)
    sentence_matrix = wv.get_sentence_matrix(clean_sentence)
    print(sentence_matrix)
    print(model.most_similar(positive=['man'], negative=['weather']))