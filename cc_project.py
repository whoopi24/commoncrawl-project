# import packages
from urllib.request import urlretrieve
import glob
import os
import re
import random
import gzip
import json
from collections import Counter
import time
from warcio import ArchiveIterator
import string
from HanTa import HanoverTagger as ht
from gensim.models import Word2Vec
import multiprocessing
import pickle

# check availability of nltk resources
import nltk
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
def get_files(crawl_name, top_lvl_domain='at', files_cnt=700):
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
def preprocess_text_corpus(crawl_dir, rm_stopwords=False):
    input_fname = os.path.join(crawl_dir, "text_corpus.txt")
    tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

    # Pre-compile regular expressions
    pattern1 = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])')
    pattern2 = re.compile(r'<[^>]+>')
    pattern3 = re.compile(r'\[([^]]+)]')

    # usage of sets for faster membership checks
    stop_words = set(nltk.corpus.stopwords.words('german'))

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
                line_tok = sent.split()     # line_tok = nltk.word_tokenize(sent)
                # remove punctuation and long 'words'
                filtered_tok = [token.strip() for token in line_tok
                                if token not in string.punctuation and len(token) < 16]
                # optional stopwords removal
                if rm_stopwords:
                    filtered_tok = [token for token in filtered_tok if token not in stop_words]
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
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_sw_removal_" + str(rm_stopwords))
    with open(pickle_fname, "wb") as save_pickle:
        pickle.dump(final_sent, save_pickle)


def train_model(crawl_dir, rm_stopwords=False):
    cores = multiprocessing.cpu_count()  # number of cores in computer

    # load preprocessed data
    pickle_fname = os.path.join(crawl_dir, "text_corpus_processed_sw_removal_" + str(rm_stopwords))
    with open(pickle_fname, "rb") as load_pickle:
        sentences = pickle.load(load_pickle)

    # ToDo: parameter tuning of word2vec arguments
    model = Word2Vec(min_count=5,
                     window=5,
                     vector_size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

    t = time.time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} minutes'.format(round((time.time() - t) / 60, 2)))

    t = time.time()
    model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} minutes'.format(round((time.time() - t) / 60, 2)))

    # save model
    model_fname = os.path.join(crawl_dir, "word2vec_sw_removal_" + str(rm_stopwords) + ".model")
    model.save(model_fname)


def evaluate_model(crawl_dir, rm_stopwords=False):
    # load model
    model_fname = os.path.join(crawl_dir, "word2vec_sw_removal_" + str(rm_stopwords) + ".model")
    model = Word2Vec.load(model_fname)

    # validate word sets of target words
    target_words = ['angreifen', 'anfassen', 'anlangen']
    for target in target_words:
        # ToDo: find out difference btw these fcts
        w1 = model.wv.most_similar(target, 10)
        w2 = model.wv.similar_by_word(target, 10)  # seems better
        print(w1)
        print(w2)


if __name__ == '__main__':
    t = time.time()
    crawl_name = 'CC-MAIN-2013-20'  # take a small crawl for testing
    top_lvl_domain = 'at'
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    rm_stopwords = False
    # get_files(crawl_name, top_lvl_domain)
    # create_text_corpus(crawl_name, top_lvl_domain)
    preprocess_text_corpus(crawl_dir, rm_stopwords)
    train_model(crawl_dir, rm_stopwords)
    evaluate_model(crawl_dir, rm_stopwords)
    print("Execution ran for", round((time.time() - t) / 60, 2), "minutes.")
