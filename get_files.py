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
import nltk
nltk.download('stopwords')
from langdetect import detect
import pandas as pd
import numpy as np


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
def get_files(crawl_name, top_lvl_domain='at'):
    # download cluster.idx file for this crawl
    path1 = 'https://data.commoncrawl.org/cc-index/collections/'
    path2 = '/indexes/'
    path_ccrawl = path1 + crawl_name + path2
    url = path_ccrawl + 'cluster.idx'
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    if not os.path.exists(crawl_dir):
        os.makedirs(crawl_dir)
    cluster_file = os.path.join(crawl_dir, "cluster.txt")
    tryDownload(url, cluster_file)

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
        filename = os.path.join(crawl_dir, file)
        tryDownload(url, filename)
        print("Successfully downloaded" + file)

    # get correct wet files
    wet_files = []
    for file in cdx_files:
        warc_files = []
        filename = os.path.join(crawl_dir, file)
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
            count_dict = Counter(warc_files)

            # only download wet files with a lot of occurrences
            for key, value in count_dict.items():
                if value >= 50:
                    key = key.replace("/warc/", "/wet/").replace("warc.gz", "warc.wet.gz")
                    key_path = key.split("/")[-1]
                    filename = os.path.join(crawl_dir, key_path)
                    if filename not in wet_files:
                        wet_files.append(filename)
                        url = "https://data.commoncrawl.org/" + key
                        print(url)
                        tryDownload(url, filename)
    print("Download of wet files finished.")


# function to create text corpus of specific crawl and for specific top-level domain
def transform_warc_to_txt(crawl_name, top_lvl_domain='at'):
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    os.chdir(crawl_dir)
    iter = 1
    for w in glob.glob("*.warc.wet.gz"):
        print("Wet file nr. ", iter)
        fname = w.replace(".warc.wet.gz", "-text.txt")
        with open(w, 'rb') as stream, open(fname, 'wt', encoding='utf-8') as f:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'conversion':
                    regex = '\.' + top_lvl_domain + '/'
                    match = re.search(regex, record.rec_headers.get_header('WARC-Target-URI'))
                    length = int(record.rec_headers.get_header('Content-Length'))
                    rec_type = record.rec_headers.get_header('Content-Type')
                    if match and length > 10000 and rec_type == "text/plain":
                        print(record.rec_headers.get_header('WARC-Target-URI'))
                        content = record.content_stream().read().decode('utf-8', errors='replace')
                        # ToDo: maybe pre-processing before saving unnecessary lines ?
                        f.write(content)
        iter += 1
    print("All wet files successfully pre-processed.")

# function to create text corpus of specific crawl and for specific top-level domain
def create_text_corpus(crawl_name, top_lvl_domain='at'):
    crawl_dir = os.path.join("S:", "msommer", crawl_name, top_lvl_domain)
    output_file = os.path.join(crawl_dir, "text_corpus.txt")
    stop_words = nltk.corpus.stopwords.words('german')
    min_stopwords = 10
    with open(output_file, 'w', encoding="utf-8") as output:
        os.chdir(crawl_dir)
        for fname in glob.glob("*-text.txt"):
            with open(fname, "rt", encoding="utf-8") as file:
                last_line = None
                for line in file:
                    sw_cnt = sum(word in stop_words for word in line.split())
                    comma_cnt = line.count(',')
                    if sw_cnt < min_stopwords or line.count('.') < 3 or len(line) < 100 or comma_cnt / sw_cnt > 1:
                        continue
                    # ToDo: remove stopwords, tokenization?
                    # ToDO: encoding problems with Umlauten -> not solvable
                    # ToDo: look into word2vec input requirements
                    # ToDo: remove "word" which are not actually words (and super long)
                    # remove duplicated sequential lines
                    if line == last_line:
                        continue
                    output.write(line)
                    last_line = line

# function to preprocess text corpus (text_corpus.txt file)
#def preprocess_text_corpus(crawl_name, top_lvl_domain='at'):

if __name__ == '__main__':
    start_time = time.time()
    crawl_name = 'CC-MAIN-2013-20'   # take a small crawl for testing
    top_lvl_domain = 'at'
    #get_files(crawl_name, top_lvl_domain)
    #transform_warc_to_txt(crawl_name, top_lvl_domain)
    create_text_corpus(crawl_name, top_lvl_domain)
    print("Execution ran for", time.time() - start_time, "seconds")
