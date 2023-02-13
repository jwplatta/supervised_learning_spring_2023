import os
import json
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

import transformers

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from tqdm import tqdm
from .constants import DATA_DIR


class SNLICorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, file_path):
        data = [json.loads(row) for row in open(file_path, 'r')]
        data = pd.DataFrame(data=data)
        data = data[data['gold_label'] != '-'].reset_index(drop=True).copy()
        self.data = data[['gold_label', 'sentence1', 'sentence2']].copy()
        self.data['combined'] = '[CLS]' + data['sentence1'] + '[SEP]' + data['sentence2'] + '[SEP]'

        self.sentences = pd.concat(
            [data['sentence1'], data['sentence2'], pd.Series(['[UNK]'])],
            axis=0
        ).reset_index(drop=True)
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.special_tokens = ['[CLS]', '[SEP]', '[UNK]']


    def __iter__(self):
        for line in self.sentences:
            # NOTE: assume there's one document per line, tokens separated by whitespace
            word_tokens = self.tokenizer.tokenize(line)

            filtered_sent = []
            for w in word_tokens:
                if w in self.special_tokens:
                    filtered_sent.append(w)
                elif not w.lower() in self.stop_words:
                    filtered_sent.append(w.lower())

            if not(filtered_sent):
                yield ['[UNK]']
            else:
                yield filtered_sent

            # filtered_sent = []
            # for w in word_tokens:
            #     if w == 'EMPTY':
            #         filtered_sent.append(w)
            #     else:
            #         filtered_sent.append(w.lower())
            # yield filtered_sent


class BertWord2VecTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        word2vec = self.model()
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')

        sent_vecs = pd.DataFrame(np.full((X.shape[0], 768), np.nan))

        for sent_idx in tqdm(range(X.shape[0])):
            sent = X.iloc[sent_idx]
            word_tokens = tokenizer.tokenize(sent)

            filtered_sent = [
                w.lower() for w in word_tokens
                if not w.lower() in stop_words and w in word2vec.wv.key_to_index.keys()
            ]

            if not(filtered_sent):
                sent_vec = word2vec.wv[['[UNK]']].mean(axis=0)
            else:
                sent_vec = word2vec.wv[filtered_sent].mean(axis=0)

            sent_vecs.iloc[sent_idx] = sent_vec

        return sent_vecs


    def fit(self, X, y=None, **fit_params):
        return self


    def model(self):
        bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        inputs = bert.embeddings.word_embeddings.weight.data
        word_embeddings = inputs.numpy()
        word2vec_model = Word2Vec(vector_size=word_embeddings.shape[1])
        word2vec_model.build_vocab_from_freq(tokenizer.vocab)
        word2vec_model.wv.vectors = word_embeddings

        return word2vec_model


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        embedding_size = 150
        word2vec = self.model()
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        special_tokens = ['[CLS]', '[SEP]', '[UNK]']

        sent_vecs = pd.DataFrame(np.full((X.shape[0], embedding_size), np.nan))

        for sent_idx in tqdm(range(X.shape[0])):
            sent = X.iloc[sent_idx]
            word_tokens = tokenizer.tokenize(sent)

            # filtered_sent = [
            #     w.lower() for w in word_tokens if not w.lower() in stop_words and w in word2vec.wv.key_to_index.keys()
            # ]
            filtered_sent = []
            for w in word_tokens:
                if w in special_tokens:
                    filtered_sent.append(w)
                elif not w.lower() in stop_words and w in word2vec.wv.key_to_index.keys():
                    filtered_sent.append(w.lower())

            if not(filtered_sent):
                sent_vec = word2vec.wv[['[UNK]']].mean(axis=0)
            else:
                sent_vec = word2vec.wv[filtered_sent].mean(axis=0)

            sent_vecs.iloc[sent_idx] = sent_vec

        return sent_vecs


    def fit(self, X, y=None, **fit_params):
        return self


    def model(self):
        word2vec_path = os.path.join(DATA_DIR, "snli_word2vec.model")
        return Word2Vec.load(word2vec_path)


class SNLIFeaturesBERT:
    def __init__(self, instances=None):
        self.word2vec_model = "snli_word2vec.model"
        self.train_data_filename = "snli_1.0_train.jsonl"
        self.test_data_filename = "snli_1.0_test.jsonl"
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = None
        self.class_labels = None
        self.class_labeler = LabelEncoder()
        self.train_corp = None
        self.test_corp = None
        self.instances = instances
        self.embedding_size = 768
        self.__load()

    def __load(self):
        word2vec_path = os.path.join(DATA_DIR, self.word2vec_model)
        train_data_path = os.path.join(DATA_DIR, self.train_data_filename)
        test_data_path = os.path.join(DATA_DIR, self.test_data_filename)

        snli_train_corp = SNLICorpus(train_data_path)
        snli_test_corp = SNLICorpus(test_data_path)

        preprocessor = ColumnTransformer(
            transformers=[
                ('combined_wv', BertWord2VecTransformer(), 0)
            ]
        )

        pipe = Pipeline([
            ("word_vectors", preprocessor),
            ("scaler", StandardScaler())
        ])

        feature_cols = ['feature_{0}'.format(idx) for idx in range(1, self.embedding_size+1)]

        X_train = pipe.fit_transform(snli_train_corp.data[['combined']])
        self.X_train = pd.DataFrame(X_train, columns=feature_cols)

        X_test = pipe.fit_transform(snli_test_corp.data[['combined']])
        self.X_test = pd.DataFrame(X_test, columns=feature_cols)

        self.class_labeler.fit(snli_train_corp.data['gold_label'])
        self.classes = self.class_labeler.classes_.tolist()
        self.class_labels = self.class_labeler.transform(self.class_labeler.classes_).tolist()

        y_train = self.class_labeler.transform(snli_train_corp.data['gold_label'])
        self.y_train = pd.Series(y_train)

        y_test = self.class_labeler.transform(snli_test_corp.data['gold_label'])
        self.y_test = pd.Series(y_test)


class SNLIData:
    def __init__(self, instances=None, random_state=279):
        self.word2vec_model = "snli_word2vec.model"
        self.train_data_filename = "snli_1.0_train.jsonl"
        self.test_data_filename = "snli_1.0_test.jsonl"
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = None
        self.class_labels = None
        self.class_labeler = LabelEncoder()
        self.train_corp = None
        self.test_corp = None
        self.instances = instances
        self.embedding_size = 150
        self.random_state = random_state
        self.__load()

    def __load(self):
        word2vec_path = os.path.join(DATA_DIR, self.word2vec_model)
        train_data_path = os.path.join(DATA_DIR, self.train_data_filename)
        test_data_path = os.path.join(DATA_DIR, self.test_data_filename)

        snli_train_corp = SNLICorpus(train_data_path)
        snli_test_corp = SNLICorpus(test_data_path)

        # if self.instances:

        # NOTE: build the word2vec model if it doesn't exist
        if os.path.exists(word2vec_path):
            word2vec = Word2Vec.load(word2vec_path)
        else:
            word2vec = Word2Vec(sentences=snli_train_corp, vector_size=self.embedding_size, epochs=10, window=10, min_count=1, workers=4)
            word2vec.save(word2vec_path)

        preprocessor = ColumnTransformer(
            transformers=[
                ('combined_wv', Word2VecTransformer(), 0)
            ]
        )

        pipe = Pipeline([
            ("word_vectors", preprocessor),
            ("scaler", StandardScaler())
        ])

        feature_cols = ['feature_{0}'.format(idx) for idx in range(1, self.embedding_size+1)]

        X_train = pipe.fit_transform(snli_train_corp.data[['combined']])
        self.X_train = pd.DataFrame(X_train, columns=feature_cols)

        X_test = pipe.fit_transform(snli_test_corp.data[['combined']])
        self.X_test = pd.DataFrame(X_test, columns=feature_cols)

        self.class_labeler.fit(snli_train_corp.data['gold_label'])
        self.classes = self.class_labeler.classes_.tolist()
        self.class_labels = self.class_labeler.transform(self.class_labeler.classes_).tolist()

        y_train = self.class_labeler.transform(snli_train_corp.data['gold_label'])
        self.y_train = pd.Series(y_train)

        y_test = self.class_labeler.transform(snli_test_corp.data['gold_label'])
        self.y_test = pd.Series(y_test)