# import packages
import glob
import gzip
import json
import multiprocessing
import os
import pickle
import random
import re
import string
import time
from collections import Counter
from urllib.request import urlretrieve
from itertools import product, combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim.models import Word2Vec
from warcio import ArchiveIterator
import nltk
import spacy

# download spacy german language models in terminal
# python -m spacy download de_core_news_sm
# python -m spacy download de_core_news_md
# python -m spacy download de_core_news_lg

# check availability of nltk resources
try:
    stop_words = nltk.corpus.stopwords.words('german')
except LookupError:
    print('Resource not found. Downloading now...')
    nltk.download('stopwords')
try:
    test = "That is a test."
    test = nltk.word_tokenize(test)
except LookupError:
    print('Resource not found. Downloading now...')
    nltk.download('punkt')


# function to count punctuation markers, stopwords and line breaks
def count_pct_and_stopwords(text, stopwords):
    pct_count = 0
    sw_count = 0
    lb_count = text.count('\n')
    line_tok = nltk.word_tokenize(text)

    for tok in line_tok:
        if tok in string.punctuation:
            pct_count += 1
        if tok in stopwords:
            sw_count += 1

    return pct_count, sw_count, lb_count


# function to recursively retry (10 times) to download a file
def tryDownload(url, filename, retries=0):
    if retries > 10:
        print("Download failed.")
        return
    try:
        urlretrieve(url, filename)
    except:
        time.sleep(1)
        tryDownload(url, filename, retries + 1)


# function to extract the year and week from the string
def extract_year_week(year_week_str):
    # Split the string by '-' and extract the relevant parts
    year_week = year_week_str.split('-')
    year = int(year_week[2])
    week = int(year_week[3])
    return f'{year}-{week}'


# function to get wet files of specific crawl and for specific top-level domain
def get_files(crawl_name, top_lvl_domain='at', files_cnt=1000):
    # download cluster.idx file for this crawl
    path1 = 'https://data.commoncrawl.org/cc-index/collections/'
    path2 = '/indexes/'
    path_ccrawl = path1 + crawl_name + path2
    url = path_ccrawl + 'cluster.idx'
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    if not os.path.exists(crawl_dir):
        os.makedirs(crawl_dir)
    cluster_file = os.path.join(crawl_dir, "cluster.txt")
    if not os.path.exists(cluster_file):
        tryDownload(url, cluster_file)
    else:
        print(f"The file '{cluster_file}' already exists.")

    # filter cdx files with top-level-domain (tld) at or de
    regex = '^' + top_lvl_domain + ','
    with open(cluster_file, "rt") as file:
        cdx_files = []
        for line in file:
            tmp = line.split("\t")
            match = re.search(regex, tmp[0])
            if match:
                if not tmp[1] in cdx_files:
                    cdx_files.append(tmp[1])

    # choose only one cdx file
    if len(cdx_files) > 1:
        random.seed(24)
        idx = random.sample(range(0, len(cdx_files)), 1)
        cdx_files = [cdx_files[i] for i in idx]

    # download cdx files
    for file in cdx_files:
        url = path_ccrawl + file
        file_path = os.path.join(crawl_dir, file)
        if not os.path.exists(file_path):
            tryDownload(url, file_path)
            print("Successfully downloaded " + file)
        else:
            print(f"The file '{file_path}' already exists.")

    # get correct wet files
    wet_files = []
    for file in cdx_files:
        warc_files = []
        filename = os.path.join(crawl_dir, file)
        iter = 0
        with gzip.open(filename, 'rt') as f:
            for line in f:
                match = re.search(regex, line)
                if match:
                    l = line.split("{\"url\":")
                    string = "{\"url\":" + l[1]
                    d = json.loads(string)
                    warc = d["filename"]
                    if int(d["length"]) > 10000:
                        warc_files.append(warc)
            # order dict by most common wet files (descending order of value)
            count_dict = Counter(warc_files).most_common()

            # only download wet files with a lot of occurrences
            for key, value in count_dict:
                if value < 50 or iter >= files_cnt:
                    break
                key = key.replace("/warc/", "/wet/").replace("warc.gz", "warc.wet.gz")
                key_path = key.split("/")[-1]
                filename = os.path.join(crawl_dir, key_path)
                if os.path.exists(filename):
                    print("File already exists.")
                    continue
                elif filename not in wet_files:
                    iter += 1
                    wet_files.append(filename)
                    url = "https://data.commoncrawl.org/" + key
                    print(url)
                    tryDownload(url, filename)
    print("Download of wet.gz files finished.")


# function to create text corpus of specific crawl and for specific top-level domain
def create_text_corpus(crawl_name, top_lvl_domain='at', files_cnt=1000):
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    output_file = os.path.join(crawl_dir, "text_corpus.txt")
    stop_words = nltk.corpus.stopwords.words('german')
    iter = 0
    with open(output_file, 'wt', encoding="utf-8") as output:
        os.chdir(crawl_dir)
        for wet_file in glob.glob("*.warc.wet.gz"):
            iter += 1
            print("Wet file nr. ", iter)
            with open(wet_file, 'rb') as stream:
                for record in ArchiveIterator(stream):
                    if record.rec_type == 'conversion':
                        regex = '\.' + top_lvl_domain + '/'
                        match = re.search(regex, record.rec_headers.get_header('WARC-Target-URI'))
                        length = int(record.rec_headers.get_header('Content-Length'))
                        rec_type = record.rec_headers.get_header('Content-Type')
                        if match and length > 10000 and rec_type == "text/plain":
                            # print(record.rec_headers.get_header('WARC-Target-URI'))
                            content = record.content_stream().read().decode('utf-8', errors='replace')
                            pct_cnt, sw_cnt, lb_cnt = count_pct_and_stopwords(content, stop_words)
                            # print(pct_cnt, sw_cnt, lb_cnt)
                            if sw_cnt == 0:
                                continue
                            elif pct_cnt / sw_cnt > 1 or lb_cnt / sw_cnt > 0.5:
                                continue
                            output.write(content)
            # ToDo: add restriction on file size
            if iter >= files_cnt:
                break
    print("Text corpus successfully created.")
    # Note: encoding problems with umlauts -> not solvable since umlauts are incorrectly encoded in source files


# function to preprocess text corpus of specific crawl and for specific top-level domain
def preprocess_text_corpus_spacy(crawl_dir, spacy_model):
    input_fname = os.path.join(crawl_dir, "text_corpus.txt")
    nlp = spacy.load(spacy_model)
    german_words = set(nlp.vocab.strings)

    # Pre-compile regular expressions
    pattern1 = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])')
    pattern2 = re.compile(r'<[^>]+>')
    pattern3 = re.compile(r'\[([^]]+)]')

    with open(input_fname, "rt", encoding="utf-8") as input_file:
        final_sent = []
        last_line = None
        t = time.time()
        iter = 0

        for line in input_file:
            iter += 1
            # print progress
            if iter % 500 == 0:
                print(iter)

            # ignore very short lines, e.g. single words
            if len(line) < 50:
                continue

            # remove URLs and HTML tags
            cleaned_line = line
            for pattern in [pattern1, pattern2, pattern3]:
                cleaned_line = pattern.sub("", cleaned_line)

            # sentence tokenization
            sentences = nltk.sent_tokenize(cleaned_line)

            # further pre-processing steps
            for sent in sentences:
                # lemmatization and remove punctuation and long 'words'
                final_line = [token.lemma_ for token in nlp(sent)
                              if token.is_alpha
                              and len(token) < 16 # depending on average German word length
                              #and token.lemma_ in german_words # ToDo: not sure about this ... sentences become weird
                              ]
                # remove duplicated sequential lines
                if final_line == last_line:
                    continue
                # remove short sentences (less than 5 words)
                if len(final_line) < 5:
                    continue
                # append list
                final_sent.append(final_line)
                last_line = final_line

                #print(final_line)

    print('Time to pre-process text: {} minutes'.format(round((time.time() - t) / 60, 2)))

    # save list of tokenized sentences as pickle
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_" + spacy_model)
    with open(pickle_fname, "wb") as save_pickle:
        pickle.dump(final_sent, save_pickle)


def preprocess_text_corpus(crawl_dir):
    # ToDo: change fname to "text_corpus.txt"
    input_fname = os.path.join(crawl_dir, "text_corpus_test.txt")
    tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

    # Pre-compile regular expressions
    pattern1 = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])')
    pattern2 = re.compile(r'<[^>]+>')
    pattern3 = re.compile(r'\[([^]]+)]')

    with open(input_fname, "rt", encoding="utf-8") as input_file:
        final_sent = []
        last_line = None
        t = time.time()
        iter = 0

        for line in input_file:
            iter += 1
            # print progress
            if iter % 500 == 0:
                print(iter)

            # ignore very short lines, e.g. single words
            if len(line) < 50:
                continue

            # remove URLs and HTML tags
            cleaned_line = line
            for pattern in [pattern1, pattern2, pattern3]:
                cleaned_line = pattern.sub("", cleaned_line)

            # sentence tokenization
            sentences = nltk.sent_tokenize(cleaned_line)

            # further pre-processing steps
            for sent in sentences:
                # word tokenization
                line_tok = sent.split()  # line_tok = nltk.word_tokenize(sent)
                # remove punctuation and long 'words'
                filtered_tok = [token.strip() for token in line_tok
                                if token not in string.punctuation and len(token) < 16]
                # lemmatization
                final_line = [lemma for (word, lemma, pos) in tagger_de.tag_sent(filtered_tok)]
                # remove duplicated sequential lines
                if final_line == last_line:
                    continue
                # append list
                final_sent.append(final_line)
                last_line = final_line

    print('Time to pre-process text: {} minutes'.format(round((time.time() - t) / 60, 2)))

    # save list of tokenized sentences as pickle
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed")
    with open(pickle_fname, "wb") as save_pickle:
        pickle.dump(final_sent, save_pickle)


def train_model(crawl_dir, spacy_model):
    cores = multiprocessing.cpu_count()  # number of cores in computer

    # load preprocessed data
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_" + spacy_model)
    with open(pickle_fname, "rb") as load_pickle:
        sentences = pickle.load(load_pickle)

    # ToDo: parameter tuning of word2vec arguments
    model = Word2Vec(min_count=5,
                     window=5,
                     # dimensionality of word vectors, mostly 100-300, more linguistic nuance -> computational cost
                     vector_size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     epochs=5,           # number of passes over the training data
                     negative=20,        # number of negative samples for noise-contrastive training
                     workers=cores - 1,
                     sg=1                # skip-gram model (default: CBOW)
                     )

    t = time.time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} minutes'.format(round((time.time() - t) / 60, 2)))

    t = time.time()
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} minutes'.format(round((time.time() - t) / 60, 2)))

    # save model
    model_fname = os.path.join(crawl_dir, "word2vec_spacy.model")
    model.save(model_fname)


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)  # set conversion for list input
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def get_w2v_output(crawl_names, top_lvl_domains, target_words, word_cnt=20):
    year_tld = list(product(crawl_names, top_lvl_domains))
    word_lists = {}
    for year, tld in year_tld:
        # load model
        c_dir = os.path.join("S:", "msommer", year, tld)
        model_fname = os.path.join(c_dir, "word2vec_spacy.model")
        model = Word2Vec.load(model_fname)

        # get nearest neighbours of target words
        for word in target_words:
            key = tuple([extract_year_week(year), tld, word])
            # https://stackoverflow.com/questions/50275623/difference-between-most-similar-and-similar-by-vector-in-gensim-word2vec
            word_lists[key] = [result[0] for result in model.wv.similar_by_word(word, word_cnt)]

    return word_lists


def calculate_jaccard_similarity(word_sets):
    years = set([key[0] for key in word_sets.keys()])
    countries = set([key[1] for key in word_sets.keys()])
    words = set([key[2] for key in word_sets.keys()])

    # ToDo: remove jaccard_results - not necessary
    jaccard_results = {'years': [], 'countries': [], 'words': []}
    jaccard_list = []

    # comparison by years
    if len(years) > 1:
        for word in words:
            for country in countries:
                year_pairs = combinations(years, 2)
                for year1, year2 in year_pairs:
                    key1 = (year1, country, word)
                    key2 = (year2, country, word)
                    if key1 in word_sets and key2 in word_sets:
                        similarity = jaccard_similarity(word_sets[key1], word_sets[key2])
                        jaccard_results['years'].append((key1, key2, similarity))
                        jaccard_list.append([year1, year2, country, country, word, word, similarity])

    # comparison by countries
    if len(countries) > 1:
        for year in years:
            for word in words:
                country_pairs = combinations(countries, 2)
                for country1, country2 in country_pairs:
                    key1 = (year, country1, word)
                    key2 = (year, country2, word)
                    if key1 in word_sets and key2 in word_sets:
                        similarity = jaccard_similarity(word_sets[key1], word_sets[key2])
                        jaccard_results['countries'].append((key1, key2, similarity))
                        jaccard_list.append([year, year, country1, country2, word, word, similarity])

    # comparison by words
    if len(words) > 1:
        for year in years:
            for country in countries:
                word_pairs = combinations(words, 2)
                for word1, word2 in word_pairs:
                    key1 = (year, country, word1)
                    key2 = (year, country, word2)
                    if key1 in word_sets and key2 in word_sets:
                        similarity = jaccard_similarity(word_sets[key1], word_sets[key2])
                        jaccard_results['words'].append((key1, key2, similarity))
                        jaccard_list.append([year, year, country, country, word1, word2, similarity])

    # create dataframe with all comparisons
    jaccard_df = pd.DataFrame(jaccard_list, columns=[
        'year1', 'year2', 'country1', 'country2', 'word1', 'word2', 'jaccard_idx'
    ])

    return jaccard_results, jaccard_df


def plot_jaccard_similarity(jaccard_df, comparison_type):
    # plot figure
    plt.figure(figsize=(10, 6))

    # check type of comparison
    type_vec = ['words', 'countries', 'years']
    if comparison_type not in type_vec:
        raise ValueError(f"'{type}' is not correct! Choose between {type_vec}!")
    elif comparison_type == 'words':
        print("Compare word sets of different target words")
        df = jaccard_df[
            (jaccard_df['year1'] == jaccard_df['year2']) &
            (jaccard_df['country1'] == jaccard_df['country2']) &
            (jaccard_df['word1'] != jaccard_df['word2'])]
        ax = sns.lineplot(
            x='year1', y='jaccard_idx', hue='country1', data=df, marker="o", palette="deep", errorbar='sd'
        )
        # specify axis labels
        ax.set(xlabel='year',
               ylabel='jaccard similarity')
        ax.legend(title='country')

    elif comparison_type == "countries":
        print("Compare word sets of different countries")    # for same target word and year
        df = jaccard_df[
            (jaccard_df['year1'] == jaccard_df['year2']) &
            (jaccard_df['country1'] != jaccard_df['country2']) &
            (jaccard_df['word1'] == jaccard_df['word2'])]
        ax = sns.lineplot(
            x='year1', y='jaccard_idx', hue='word1', data=df, marker="o", palette="deep", errorbar='sd'
        )
        # specify axis labels
        ax.set(xlabel='year',
               ylabel='jaccard similarity')
        ax.legend(title='target words')

    elif comparison_type == "years":
        print("Compare word sets of different years")        # for same target word and country
        df = jaccard_df[
            (jaccard_df['year1'] != jaccard_df['year2']) &
            (jaccard_df['country1'] == jaccard_df['country2']) &
            (jaccard_df['word1'] == jaccard_df['word2'])]
        ax = sns.lineplot(
            x='word1', y='jaccard_idx', hue='country1', data=df, marker="o", palette="deep", errorbar='sd'
        )
        # specify axis labels
        ax.set(xlabel='target words',
               ylabel='jaccard similarity')
        ax.legend(title='country')

    plt.title(f'Jaccard Similarity ({comparison_type.capitalize()} Comparisons)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    t = time.time()
    crawl_name = 'CC-MAIN-2013-20'  # take a small crawl for testing
    top_lvl_domain = 'de'
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)

    # check target words
    spacy_model = 'de_core_news_md'
    nlp = spacy.load(spacy_model)
    vocab = set(nlp.vocab.strings)
    target_words = ['angreifen', 'anfassen', 'anlangen']
    for word in target_words:
        if word not in vocab:
            raise ValueError(f"'{word}' is not in spacy '{spacy_model}' vocabulary!")

    get_files(crawl_name, top_lvl_domain)
    create_text_corpus(crawl_name, top_lvl_domain)
    # preprocess_text_corpus(crawl_dir)
    preprocess_text_corpus_spacy(crawl_dir, spacy_model)
    train_model(crawl_dir, spacy_model)

    # get word sets
    # ToDo: try word_cnt = 10,20,50,100
    #output = get_w2v_output(crawl_names=['CC-MAIN-2013-20'], top_lvl_domains=['de', 'at'],
    #                        target_words=['angreifen', 'anfassen', 'anlangen'], word_cnt=10)

    # calculate jaccard index
    #results, jaccard_df = calculate_jaccard_similarity(output)

    # Plot for 'years' comparison
    #plot_jaccard_similarity(jaccard_df, 'years')

    # Plot for 'countries' comparison
    #plot_jaccard_similarity(jaccard_df, 'countries')

    # Plot for 'words' comparison
    #plot_jaccard_similarity(jaccard_df, 'words')

    print("Execution ran for", round((time.time() - t) / 60, 2), "minutes.")

    # tests
    # example word sets dictionary
    # word_sets = {
    #    (2020, 'country1', 'word1'): ['a', 'b', 'c'],
    #    (2020, 'country1', 'word2'): ['a', 'b', 'e'],
    #    (2020, 'country2', 'word1'): ['a', 'd', 'f'],
    #    (2021, 'country1', 'word1'): ['a', 'c', 'g'],
    #    (2021, 'country2', 'word1'): ['a', 'b', 'h'],
    # }

    # Compute Jaccard similarities and get the DataFrame
    # jaccard_results, jaccard_df = calculate_jaccard_by_dimension(word_sets)
    # print(jaccard_df)