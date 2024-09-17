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
from itertools import product

# check availability of nltk resources
import nltk
import spacy_udpipe
import spacy

# download spacy german language models in terminal
# python -m spacy download de_core_news_sm
# python -m spacy download de_core_news_md
# python -m spacy download de_core_news_lg

from spacy.tokenizer import Tokenizer
# from HanTa import HanoverTagger as ht
from gensim.models import Word2Vec
from warcio import ArchiveIterator

# download German spacy model
spacy_udpipe.download("de")

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


# import pandas as pd
# import numpy as np


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
            if iter >= files_cnt:
                break
    print("Text corpus successfully created.")
    # Note: encoding problems with umlauts -> not solvable since umlauts are incorrectly encoded in source files


# function to preprocess text corpus of specific crawl and for specific top-level domain
def preprocess_text_corpus_udpipe(crawl_dir):
    # ToDo: change fname to "text_corpus.txt"
    input_fname = os.path.join(crawl_dir, "text_corpus_test.txt")

    # load German spacy model
    nlp = spacy_udpipe.load("de")

    with open(input_fname, "rt", encoding="utf-8") as input_file:
        final_sent = []
        t = time.time()
        text = input_file.read()
        n = len(text)

        # presegmentation (List[str])/pretokenization (List[List[str]])
        # sentences = nltk.sent_tokenize(text)  -> doesnt work

        # ValueError: [E088] Text of length 2684477 exceeds maximum of 1000000.
        # The parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input.
        # This means long texts may cause memory allocation errors. If you're not using the parser or NER,
        # it's probably safe to increase the `nlp.max_length` limit. The limit is in number of characters,
        # so you can check whether your inputs are too long by checking `len(text)`.
        if n > 1000000:
            nlp.max_length = n # or preprocessing

        # use udpipe model
        doc = nlp(text)

        for token in doc:
            #if token.is_alpha and len(token) < 16:
            #    filtered_tok = [token.strip() for token in line_tok
            #                    if token.is_alpha and len(token) < 16]

            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha,
                  token.is_stop)
            # append list
            final_sent.append(token)

    print('Time to pre-process text: {} minutes'.format(round((time.time() - t) / 60, 2)))

    # save list of tokenized sentences as pickle
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_udpipe_" + str(rm_stopwords))
    with open(pickle_fname, "wb") as save_pickle:
        pickle.dump(final_sent, save_pickle)


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
                              and len(token) < 16
                              and token.lemma_ in german_words # ToDo: not sure about this ... sentences become weird
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
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_spacy")
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


def train_model(crawl_dir):
    cores = multiprocessing.cpu_count()  # number of cores in computer

    # load preprocessed data
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_spacy")
    with open(pickle_fname, "rb") as load_pickle:
        sentences = pickle.load(load_pickle)

    #print(sentences)
    #raise NotImplementedError()

    # ToDo: parameter tuning of word2vec arguments
    model = Word2Vec(min_count=5,
                     window=5,
                     vector_size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
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


def evaluate_model(crawl_dir):
    # load model
    model_fname = os.path.join(crawl_dir, "word2vec_spacy.model")
    model = Word2Vec.load(model_fname)
    # check vocab
    # print(model.wv.key_to_index)

    # validate word sets of target words
    target_words = ['angreifen', 'anfassen', 'anlangen']
    for target in target_words:
        # ToDo: find out difference btw these fcts - https://stackoverflow.com/questions/50275623/difference-between-most-similar-and-similar-by-vector-in-gensim-word2vec
        # ToDo: mehr Woerter verwenden - 10,20,50,100 ausprobieren
        # w1 = model.wv.most_similar(target, 10)
        w2 = model.wv.similar_by_word(target, 20)  # seems better
        # print(w1)
        print(w2)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union #if union != 0 else 0


def compare_w2v_models(crawl_names, top_lvl_domains, target_words, word_cnt=20):
    combinations = list(product(crawl_names, top_lvl_domains))
    jaccard_results = {}

    for i in range(len(combinations)):
        for j in range(i + 1, len(combinations)):
            comb1 = combinations[i]
            comb2 = combinations[j]
            c_dir1 = os.path.join("S:", "msommer", c_name, tld)
            c_dir2 = os.path.join("S:", "msommer", c_name, tld)

            # load model
            model_fname1 = os.path.join(c_dir1, "word2vec_spacy.model")
            model1 = Word2Vec.load(model_fname1)
            model_fname2 = os.path.join(c_dir2, "word2vec_spacy.model")
            model2 = Word2Vec.load(model_fname2)
            set1 = {model1.wv.similar_by_word(word, word_cnt) for word in target_words}
            set2 = {model2.wv.similar_by_word(word, word_cnt) for word in target_words}
            jaccard = jaccard_similarity(set1, set2)
            key1 = f"{comb1[0]}_{comb1[1]}"
            key2 = f"{comb2[0]}_{comb2[1]}"
            jaccard_results[(key1, key2)] = jaccard

    return jaccard_results

def compare_w2v_models(type, crawl_names, top_lvl_domains, target_words, word_cnt=20):
    # check type of comparison
    type_vec = ['word', 'country', 'time']
    if type not in type_vec:
        raise ValueError(f"'{type}' is not correct! Choose between {type_vec}!")
    elif type == 'word':
        print("Compare word sets of different target words for same crawl")
        for c_name in crawl_names:
            for tld in top_lvl_domains:
                c_dir = os.path.join("S:", "msommer", c_name, tld)

                # load model
                model_fname = os.path.join(c_dir, "word2vec_spacy.model")
                model = Word2Vec.load(model_fname)

                set1 = [a[0] for a in model.wv.similar_by_word('angreifen', word_cnt)]
                set2 = [a[0] for a in model.wv.similar_by_word('anfassen', word_cnt)]
                set3 = [a[0] for a in model.wv.similar_by_word('anlangen', word_cnt)]
                # ToDo: save information of words, years and countries
                j1 = jaccard_similarity(set1, set2)
                j2 = jaccard_similarity(set2, set3)
                j3 = jaccard_similarity(set3, set1)

    elif type == "country":
        print("Compare word sets of different countries for same target word and year")
        model_cnt = len(crawl_names) * len(top_lvl_domains)
        models = list()
        for c_name in crawl_names:
            for t_word in ['angreifen', 'anfassen', 'anlangen']:
                for tld in top_lvl_domains:
                    c_dir = os.path.join("S:", "msommer", c_name, tld)

                    # load model
                    model_fname = os.path.join(c_dir, "word2vec_test.model")
                    model = Word2Vec.load(model_fname)

                    set1 = model.wv.similar_by_word(t_word, word_cnt)
                    set2 = model.wv.similar_by_word(t_word, word_cnt)

    elif type == "time":
        print("Compare word sets of different years for same target word and country")


if __name__ == '__main__':
    t = time.time()
    crawl_name = 'CC-MAIN-2013-20'  # take a small crawl for testing
    top_lvl_domain = 'de'
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)

    # check target words
    spacy_model = 'de_core_news_lg'
    nlp = spacy.load(spacy_model)
    vocab = set(nlp.vocab.strings)
    target_words = ['angreifen', 'anfassen', 'anlangen']
    for word in target_words:
        if word not in vocab:
            raise ValueError(f"'{word}' is not in spacy '{spacy_model}' vocabulary!")

    # get_files(crawl_name, top_lvl_domain)
    # create_text_corpus(crawl_name, top_lvl_domain)
    # preprocess_text_corpus(crawl_dir)
    #preprocess_text_corpus_spacy(crawl_dir, spacy_model)
    # preprocess_text_corpus_udpipe(crawl_dir)
    #train_model(crawl_dir)
    #evaluate_model(crawl_dir)

    # compare several models
    compare_w2v_models(type='word',
                       crawl_names=['CC-MAIN-2013-20'],
                       top_lvl_domains=['de'],
                       target_words=['angreifen', 'anfassen', 'anlangen'],
                       word_cnt=20)
    print("Execution ran for", round((time.time() - t) / 60, 2), "minutes.")

    # test Jaccard index
    set_a = {"Geeks", "for", "Geeks", "NLP", "DSc"}
    set_b = {"Geek", "for", "Geeks", "DSc.", 'ML', "DSA"}
    similarity = jaccard_similarity(set_a, set_b)
    print("Jaccard Similarity:", similarity)
