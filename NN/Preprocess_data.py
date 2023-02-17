import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from itertools import islice
from googletrans import Translator, constants
import csv
from nltk.stem import WordNetLemmatizer


EMPTY_TEXTS_IDX = [14360, 39723]


class GlobalMethods:
    def __init__(self):
        pass

    @staticmethod
    def my_tokenizer(text_to_tokenize):
        words = text_to_tokenize.replace('\n', ' ').strip().lower()
        words = re.sub(r'\W+|\d+', ' ', words).split()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        list_of_words_no_stopwords = []
        for w in words:
            if w not in stop_words:
                list_of_words_no_stopwords.append(lemmatizer.lemmatize(w))
        return list_of_words_no_stopwords

    @staticmethod
    def tokenize_data(data):
        my_list = []
        for _, text in enumerate(data):
            if type(text) != str:
                my_list.append('')
                continue
            my_list.append(GlobalMethods.my_tokenizer(text))
        return my_list

    @staticmethod
    def count_ngrams(text, dictionary):
        return [text.count(d) for d in dictionary]

    @staticmethod
    def translate_texts(texts, path='translated.csv'):
        translator = Translator()
        translated = []
        fp = open(path, 'w', encoding='utf-8', newline='')
        fieldnames = ['idx', 'text']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx, text in enumerate(texts):
            translation = translator.translate(text)
            if idx % 100 == 0:
                print(idx)
                print(translation.text)
                print('\n----\n')
            if idx % 500 == 0:
                fp.flush()
            writer.writerow({'idx': idx + 1, 'text': translation.text})

            translated.append(translation.text)
        fp.close()
        return translated

    @staticmethod
    def read_translated_texts(path):
        df = pd.read_csv(path)
        return list(df.iloc[:, 1])

    @staticmethod
    def flatten_list(my_list):
        flat_list = [txt for sublist in my_list for txt in sublist]
        return flat_list

    @staticmethod
    def merge_dicts(d1, d2, d3):
        merged_dict1 = d1.copy()
        merged_dict1.update(d2)
        merged_dict2 = merged_dict1.copy()
        merged_dict2.update(d3)
        return merged_dict2

    @staticmethod
    def create_word_frequencies(word_list):
        dictionary = {}
        for w in word_list:
            dictionary[w] = dictionary.get(w, 0) + 1
        return dictionary


class PreprocessTrainData:
    def __init__(self, path='train_data.csv'):
        self.translator = Translator()
        train_data = pd.read_csv(path)
        train_data.drop(EMPTY_TEXTS_IDX[0], axis=0, inplace=True)
        train_data.drop(EMPTY_TEXTS_IDX[1], axis=0, inplace=True)
        self.train_languages = list(train_data.iloc[:, 0])
        self.train_text = list(train_data.iloc[:, 1])
        self.train_text_words = GlobalMethods.tokenize_data(self.train_text)
        self.train_text_translated = GlobalMethods.read_translated_texts('translated_train.csv')
        self.train_text_translated_words = GlobalMethods.tokenize_data(self.train_text_translated)
        self.train_labels = list(train_data.iloc[:, 2])
        self.unique_languages = list(np.unique(self.train_languages))
        self.unique_labels = list(np.unique(self.train_labels))
        self.dictionary = self.create_dictionary()
        self.X_train = self.create_features_df()
        self.y_train = self.create_labels_df()

    def create_dialect_dictionary(self, dialect):
        dialect_text = []
        for idx, label in enumerate(self.train_labels):
            if label == dialect:
                dialect_text.append(self.train_text_translated_words[idx])
        freq = GlobalMethods.create_word_frequencies(GlobalMethods.flatten_list(dialect_text))
        sorted_dict = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        slice_of_dict = dict(islice(sorted_dict.items(), 1000))
        return slice_of_dict

    def create_dictionary(self):
        dialect_dict = {}
        for dialect in self.unique_labels:
            dialect_dict[dialect] = self.create_dialect_dictionary(dialect)
        dict1 = dialect_dict[self.unique_labels[0]]
        dict2 = dialect_dict[self.unique_labels[1]]
        dict3 = dialect_dict[self.unique_labels[2]]
        merged_dict = GlobalMethods.merge_dicts(dict1, dict2, dict3)
        return list(set(list(merged_dict)))

    def create_features_df(self):
        counter = [GlobalMethods.count_ngrams(word, self.dictionary) for word in self.train_text_translated_words]
        X_train = pd.DataFrame(counter, columns=self.dictionary)
        X_train.columns = ['ngram_' + str(col) for col in list(range(len(X_train.columns)))]
        return X_train

    def create_labels_df(self):
        y_train = pd.DataFrame(self.train_labels, columns=['label'])
        y_train = y_train.replace(self.unique_labels[0], 0)
        y_train = y_train.replace(self.unique_labels[1], 1)
        y_train = y_train.replace(self.unique_labels[2], 2)
        return y_train

    def save_csv(self):
        self.X_train.to_csv('train_features.csv', index=False)
        self.y_train.to_csv('train_labels.csv', index=False)


class PreprocessTestData:
    def __init__(self, dictionary, path='test_data.csv'):
        test_data = pd.read_csv(path)
        self.test_text = list(test_data.iloc[:, 0])
        self.test_text_translated = GlobalMethods.read_translated_texts('translated_test.csv')
        self.test_text_translated_words = GlobalMethods.tokenize_data(self.test_text_translated)
        self.dictionary = dictionary
        self.X_test = self.create_features_df()

    def create_features_df(self):
        counter = [GlobalMethods.count_ngrams(word, self.dictionary) for word in self.test_text_translated_words]
        X_test = pd.DataFrame(counter, columns=self.dictionary)
        X_test.columns = ['ngram_' + str(col) for col in list(range(len(X_test.columns)))]
        return X_test

    def save_csv(self):
        self.X_test.to_csv('test_features_fin.csv', index=False)


def my_main():
    trainData = PreprocessTrainData()
    trainData.save_csv()
    testData = PreprocessTestData(trainData.create_dictionary())
    testData.save_csv()


my_main()
