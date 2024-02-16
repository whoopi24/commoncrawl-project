# import packages
from urllib.request import urlretrieve
import time
import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to recursively retry (10 times) to download a file
def tryDownload(url, filename, retries=0):
    if retries > 10:
        print("Download failed.")
        return
    try:
        urlretrieve(url, filename)
    except:
        time.sleep(1)
        tryDownload(url, filename, retries+1)

# function to create text corpus of specific crawl and for specific top-level domain
def get_files(crawl_name, tld='^at,'):
    # download cluster.idx file for this crawl
    path1 = 'https://data.commoncrawl.org/cc-index/collections/'
    path2 = '/indexes/cluster.idx'
    path_ccrawl = path1 + crawl_name + path2
    url = path_ccrawl + 'cluster.idx'
    crawl_dir = os.path.join("S:", "msommer", crawl_name)
    if not os.path.exists(crawl_dir):
        os.makedirs(crawl_dir)
    filename = os.path.join(crawl_dir, "cluster.txt")
    tryDownload(url, filename)

    # filter cdx files with top-level-domain (tld) at or de
    with open(filename, "rt") as file:
        cdx_files = []
        for line in file:
            tmp = line.split("\t")
            match = re.search(tld, tmp[0])
            if match:
                if not tmp[1] in cdx_files:
                    cdx_files.append(tmp[1])

    # choose and download cdx files, e.g. cdx-00003.gz -> way too much for de
    if len(cdx_files) > 3:
        idx = random.sample(range(0, len(cdx_files)), 3)
        cdx_files = [cdx_files[i] for i in idx]

    for file in cdx_files:
        print(file)
        url = path_ccrawl + file
        filename = os.path.join(crawl_dir, file)
        tryDownload(url, filename)


