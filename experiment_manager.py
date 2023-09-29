from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from itertools import chain
import json
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os
import re
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances, euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import statistics

if os.path.exists(os.path.join(os.getcwd(), 'experiments')):
    pass
else:
    os.mkdir('experiments')
class Corpus:
    def __init__(self, name:str, corpus_path:str):
        if not os.path.exists(corpus_path):
            raise ValueError('The path to the corpus seems to be invalid')
        self.name = name
        self.corpus = corpus_path

    def load(self) -> dict:
        files = [filename for filename in os.listdir(self.corpus) if filename.endswith('.txt')]
        documents = {}
        for filename in files:
            if filename.endswith('.txt'):
                title = re.sub(r'.txt', '', filename)
                with open(os.path.join(self.corpus, filename), 'r', encoding='utf-8') as f:
                    documents[title] = f.read()
        return documents

    def get_metadata(self, save_json=None):
        metadata = {}
        for index, file in enumerate(os.listdir(self.corpus)):
            title = re.sub(r'.txt', '', file)
            metadata[f'Text {index+1}'] = {'title': title, 'file': file}
        if save_json:
            with open(f'{self.name}_metadata.json', 'w') as corpus_json:
                json.dump(metadata, corpus_json, indent=4)
        return metadata

class Tokenizer(ABC):
    """
    This abstract class allows the creation of other tokenizers
    - Input: corpus:dict
    - Output: tokenized_corpus:dict
    """
    @abstractmethod
    def tokenize_corpus(self, corpus:dict) -> dict:
        pass

class Corpus_WordTokenizer(Tokenizer):
    """
    NLTK-based word tokenizer which works with entire corpus
    """
    def __init__(self, lowercased:bool, stop_words:bool):
        self.lowercase = lowercased
        self.remove_stopwords = stop_words

    def tokenize_corpus(self, corpus:dict) -> dict:
        tokenized_corpus = {}
        for title, text in corpus.items():
            tokens = nltk.word_tokenize(text, 'german')
            if self.lowercase == True:
                tokens = [token.lower() for token in tokens]
            if self.remove_stopwords == True:
                stop_words = set(stopwords.words('german'))
                tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
            tokenized_corpus[title] = tokens
        return tokenized_corpus

class Corpus_RegexpTokenizer(Tokenizer):
    """
    NLTK-based regexp tokenizer which works with entire corpus
    """
    def __init__(self, lowercased=True, stop_words=True):
        self.lowercase = lowercased
        self.remove_stopwords = stop_words
        self.corpus_tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

    def tokenize_corpus(self, corpus:dict) -> dict:
        tokenized_corpus = {}
        for title, text in corpus.items():
            tokens = self.corpus_tokenizer.tokenize(text)
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            if self.remove_stopwords:
                stop_words = set(stopwords.words('german'))
                tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
            tokenized_corpus[title] = tokens
        return tokenized_corpus

class DistanceCalculator(ABC):
    """
    Abstract base class, which allows the creation of many distance metrics
    Input: tokenized_corpus:dict
    Output: distance matrix as pandas Dataframe
    """
    @abstractmethod
    def most_frequent_words(self):
        pass
    @abstractmethod
    def vectorize(self):
        pass

    @abstractmethod
    def distance_matrix(self):
        pass

    @abstractmethod
    def plot_dendrogram(self):
        pass

# class Corpus_TFIDF_Distance(DistanceCalculator):
#     def __init__(self, mfw:int, distance:str, corpus:dict):
#         self.mfw = mfw
#         self.distance = distance
#         self.corpus = corpus
#         self.most_frequent_terms = self.most_frequent_words()
#         self.vectorized_data = self.vectorize()
#         self.dist_matrix = self.distance_matrix()
#
#     def most_frequent_words(self):
#         term_frequencies = Counter(list(chain.from_iterable(self.corpus.values())))
#         mft = term_frequencies.most_common(self.mfw)
#         df_mft = pd.DataFrame(mft, columns=['Term', 'Frequency'])
#         return df_mft
#
#     def vectorize(self):
#         documents_tfidf = pd.DataFrame(columns=['Title'] + self.most_frequent_terms['Term'].tolist())
#         n = len(self.corpus)
#         for key, value in self.corpus.items():
#             row = [key]
#             for term in self.most_frequent_terms['Term'].tolist():
#                 rel_tf = value.count(term) / len(value)
#                 df = sum(1 for doc in self.corpus.values() if term in doc)
#                 idf = math.log(n/(df+1))
#                 tf_idf = rel_tf*idf
#                 row.append(tf_idf)
#             documents_tfidf.loc[len(documents_tfidf)] = row
#         return documents_tfidf
#
#     def distance_matrix(self):
#         document_titles = self.vectorized_data['Title'].tolist()
#         tf_idf_values = self.vectorized_data.drop(columns=['Title']).values
#         if self.distance == 'manhattan':
#             distance_matrix = manhattan_distances(tf_idf_values)
#         if self.distance == 'euclidean':
#             distance_matrix = euclidean_distances(tf_idf_values)
#         if self.distance == 'cosine':
#             distance_matrix = cosine_distances(tf_idf_values)
#         df_distance_matrix = pd.DataFrame(distance_matrix, index=document_titles, columns=document_titles)
#
#         return df_distance_matrix
#
#     def plot_dendrogram(self):
#         g = sns.clustermap(self.dist_matrix, cmap='mako',
#                            method='single', row_cluster=True,
#                            col_cluster=False, figsize=(8, 6))
#         # Show the plot
#         plt.title(f'TF-IDF. mfw: {self.mfw}, distance: {self.distance}')
#         plt.show()

class Corpus_BurrowsDistance(DistanceCalculator):
    def __init__(self, mfw:int, corpus:dict, distance:str):
        self.mfw = mfw
        self.corpus = corpus
        self.most_frequent_terms = self.most_frequent_words()
        self.vectorized_data = self.vectorize()
        self.distance = distance
        self.dist_matrix = self.distance_matrix()

    def most_frequent_words(self):
        term_frequencies = Counter(list(chain.from_iterable(self.corpus.values())))
        if self.mfw > len(term_frequencies):
            raise IndexError(f'The mfw of {self.mfw} is bigger than the length of each text in corpus')
        mft = term_frequencies.most_common(self.mfw)
        df_mft = pd.DataFrame(mft, columns=['Term', 'Frequency'])
        return df_mft

    def vectorize(self):
        documents_zscores = pd.DataFrame(columns=['Title'] + list(self.most_frequent_terms['Term']))
        n = len(self.corpus)
        for key, value in self.corpus.items():
            row = [key]
            for term in self.most_frequent_terms['Term'].tolist():
                if term in value:
                    rel_tf = value.count(term) / len(value)
                else:
                    rel_tf = 0
                rel_tf_corpus = [(doc.count(term) / len(doc)) for doc in self.corpus.values()]
                mean = sum(rel_tf_corpus) / n
                std = statistics.stdev(rel_tf_corpus)
                z_score = (rel_tf - mean) / std
                row.append(z_score)
            documents_zscores.loc[len(documents_zscores)] = row
        return documents_zscores

    def distance_matrix(self):
        document_titles = self.vectorized_data['Title'].tolist()
        z_values = self.vectorized_data.drop(columns=['Title']).values
        if self.distance == 'manhattan':
            distance_matrix = manhattan_distances(z_values)
        if self.distance == 'euclidean':
            distance_matrix = euclidean_distances(z_values)
        if self.distance == 'cosine':
            distance_matrix = cosine_distances(z_values)
        df_distance_matrix = pd.DataFrame(distance_matrix, index=document_titles, columns=document_titles)

        return df_distance_matrix

    def plot_dendrogram(self):
        g = sns.clustermap(self.dist_matrix, cmap='mako',
                           method='single', row_cluster=True,
                           col_cluster=False, figsize=(8,6))
        # Show the plot
        plt.title(f'Burrows Delta. mfw: {self.mfw}, distance: {self.distance}')
        plt.show()


class Experiment:
    def __init__(self, experiment_name, corpus_name:str, corpus_path:str, tokenizer:str,
                 stop_words:bool, lowercased:bool,
                 distance_methods:list, mfw_range:list,
                 n_cluster:int):
        self.corpus_name = corpus_name
        self.experiment_name = experiment_name
        self.corpus = Corpus(corpus_name, corpus_path)
        self.tokenizer = tokenizer # word or regexp
        self.stopwords = stop_words
        self.lowercase = lowercased
        self.mfw_range = mfw_range
        self.distance_methods = distance_methods
        self.n_cluster = n_cluster

    def run(self):
        data = self.corpus.load()
        save_corpus_metadata = self.corpus.get_metadata(save_json=True)
        if self.tokenizer == 'word':
            tokenizer = Corpus_WordTokenizer(self.lowercase, self.stopwords)
        if self.tokenizer == 'regexp':
            tokenizer = Corpus_RegexpTokenizer(self.lowercase, self.stopwords)
        tokenized_data = tokenizer.tokenize_corpus(data)

        # Dataframe
        results = []
        for distance in self.distance_methods:
            for mfw in range(self.mfw_range[0], self.mfw_range[1]+1, 10): # Default step is 10
                measuring = Corpus_BurrowsDistance(mfw, tokenized_data, distance)
                distance_matrix = measuring.distance_matrix()
                clustering = AgglomerativeClustering(self.n_cluster, linkage='single', metric='precomputed')
                cluster_labels = clustering.fit_predict(distance_matrix)
                silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                results.append({'Distance Method': distance, 'MFW': mfw, 'Silhouette Score': silhouette})

        result_df = pd.DataFrame(results)
        result_df.to_csv(os.path.join('experiments', f'{self.experiment_name}_results_table.csv'), index=False)

        # Set options for plot
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=result_df, x='MFW', y='Silhouette Score', hue='Distance Method', marker="o")
        plt.xlabel('Features (MFW)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Experiment Results')
        plt.legend(title='Distance Method')
        plt.savefig(os.path.join('experiments', f'{self.experiment_name}_silhouette_scores.png'))

        # Plot result and show Dataframe
        plt.show()
        return result_df

    def save_metadata(self):
        # Create a datastructure
        experiment_metadata = {'Experiment': self.experiment_name,
                               'Corpus': self.corpus_name,
                               'Tokenizer': self.tokenizer,
                               'Lowercase': self.lowercase,
                               'Stop words': self.stopwords,
                               'MFW': ', '.join(map(str, self.mfw_range)),
                               'Distance Methods': ', '.join(self.distance_methods),
                               'Predefined Clusters': self.n_cluster,
                               'Time': datetime.now().strftime('%d.%m.%Y, %H:%M:%S')}

        # Save as json file
        json_path = os.path.join('experiments', f'{self.experiment_name}_metadata.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(experiment_metadata, json_file, indent=4)

        return experiment_metadata