import gensim
import numpy as np


pretrained_word_vector_binary = '/Users/saur6410/Google Drive/VT/Old/Thesis/Source/ThesisPython/data/GoogleNews-vectors-negative300.bin'



def write_to_file(lv,file):
    with open(file,"w") as f:
        for l,v in lv:
            f.write(str(l) + "," + ','.join((str(x)for x in np.nditer(v))) + "\n")

class word2vec:
    model = None
    models = {}

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

    def avg_feature_vector(self, words, model, num_features, index2word_set):
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

    def read_from_file(self, name, model):
        with open(name,"r") as f:
            lines = f.readlines()
            tweets_only = map(lambda line: line.split(',')[1].rstrip().split(" "), lines)
            lables_only = map(lambda line: line.split(',')[0].rstrip(), lines)
            vecs = map(lambda t: self.avg_feature_vector(t, model, 300, model.index2word),tweets_only)
            return zip(lables_only,vecs)


if __name__ == "__main__":
    model = word2vec().get_model()
    # you can find the terms that are similar to a list of words and different from
    # another list of words like so
    print(model.most_similar(positive=['hurricane'], negative=['isaac']))

    # you can also get the vector for a specific word by doing
    print(model['hurricane'])

    # you can ask for similarity by doing
    print(model.similarity('hurricane', 'shooting'))

    print("done")