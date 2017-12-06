import gensim
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string


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

class word2vec:
    model = None
    models = {}
    @staticmethod
    def get_model():
        if not word2vec.model:
            model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            model.init_sims(replace=True)  # we are no longer training the model so allow it to trim memory
            word2vec.model = model

        return word2vec.model

    @staticmethod
    def get_model_from_file(name):
        if not name in word2vec.models:
            sentences = []
            with open(name,"r") as f:
                sentences = map(lambda x: word2vec.extract_sentence(x).split(), f.readlines())
            file_model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1, window=5)
            word2vec.models[name] = file_model.wv
        return word2vec.models[name]
    @staticmethod
    def get_model_from_sentences(sentences):
            file_model = gensim.models.Word2Vec(sentences=sentences, size=100, min_count=1, window=5)
            return file_model.wv

    @staticmethod
    def extract_sentence(line):
        lineContent = line.split(',')
        return (lineContent[1] if len(lineContent) > 1 else lineContent[0])



def clean_string(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(s)
    tokens = [w.lower() for w in tokens]

    stripped = [w.translate(None, string.punctuation) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = list(map(wordnet_lemmatizer.lemmatize, words))
    return words

def read_csv(filename):
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
                sentences.append(clean_string(desc))
                body = cells[8]
                sentences.append(clean_string(body))
        t = []
        for sentence in sentences:
            if len(sentence) != 0:
                t.append(sentence)
        return t


if __name__ == "__main__":
    sentences = read_csv("Kaggle-str-process.csv")
    model = word2vec.get_model_from_sentences(sentences)
    print(model.most_similar(positive=['man'], negative=['weather']))