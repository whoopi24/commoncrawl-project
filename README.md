# 194.147 Interdisciplinary Project in Data Science <hr>
## An analysis of the development of the German touch verbs 'anfassen', 'angreifen', 'anlangen' with text data from `Common Crawl`.
### A comparison between Austria and Germany.
by Marina Sommer | 11778902 | TU Wien | October 24, 2024 <hr>


### Introduction <hr>
This project consists of the following files:
- train.py
- evaluation.py
- requirements.txt
- ReadMe.md

If you want to know how to run this project in `Python`, go to the end of this file.

Since natural language develops at any time, the aim of this project is to find out if the usage
of the German touch verbs “anfassen”, “angreifen” and “anlangen” has changed over the last
decade. This analysis includes a comparison of two varieties of German, one spoken in Austria
and the other in Germany. The text data has been collected from `Common Crawl` and was used to
train one word embedding model per time period and variety. The word sets of the semantically
related words to the target words have been compared using the “Jaccard index” in two different
ways. Ultimately, broader insights could be obtained in relation to the findings of Ahlers and
Fink (2017) due to the time aspect involved in the evaluation. The results show that the word
sets for these verbs have changed over time, with more variation in Austria, and that the word
sets of “angreifen” are linked to actions of attack as anticipated by Ahlers and Fink (2017).
Despite facing technical limitations in data collection and preparation, this exploratory study
lays the groundwork for future research utilizing `Common Crawl` data to examine linguistic trends
over extended time periods.

### Data Collection <hr>
Included functions:
- get_files(crawl_dir, crawl_name, top_lvl_domain='at', files_cnt=500, skip=False)
- tryDownload(url, filename, retries=0)

In this section, all necessary files are downloaded, e.g. `cluster.idx`, at most two gzip-compressed `.cdx` files (amount
can be modified) and the `.wet` files. Due to occasional server problems, a file might cannot be downloaded at the 
first attempt. The function `tryDownload()` tries to download the file ten times until it is skipped. The argument 
`skip` can be used to skip the first `.cdx` file when there is more than one available.

### Data Preparation <hr>
Included functions:
- create_text_corpus(crawl_dir, top_lvl_domain='at', files_cnt=1000)
- preprocess_text_corpus_spacy(crawl_dir, spacy_model)
- count_pct_and_stopwords(text, stopwords)

The function `create_text_corpus()` creates a file called `text_corpus.txt`, which consists of the relevant text data 
of all downloaded `.wet` files. The other function, `preprocess_text_corpus_spacy()`, is used for data preparation. 
This includes removing very short lines, as well as URLs and HTML tags, sentence tokenization,
word tokenization with lemmatization, removal of every punctuation mark and every token exceeding 15 characters
as well as duplicated sequential lines and very short sentences with less than five words. For German 
lemmatization, the Python package `spaCy` provides various models, which differ in type and size. For this project, the model `de_core_news_md`, which
is of medium size, has been used. For future needs, the `spaCy` model can easily be changed.
This entire procedure of creating a pre-processed text corpus has to be done for every crawl and
every variety, i.e. top-level domain. The helper function `count_pct_and_stopwords()` is used to count stopwords,
punctuation marks and line breaks.

### Model Training <hr>
Included functions:
- train_model(crawl_dir, spacy_model)

The main part of this function is the call of `Word2Vec()` from the `gensim` package. Each text corpus has been used 
to train one word embedding model. The model needs a broad text corpus
as input and outputs a vector representation of each word in the vocabulary of the training
data. Hence, it is possible to predict the context from one specific word by looking at its nearest
neighbors in the vector space. The model is saved as `.model` object.


### Evaluation <hr>
Included functions:
- get_w2v_output(data_path, crawl_names, top_lvl_domains, target_words, spacy_model, word_cnt=100)
- calculate_jaccard_similarity(word_sets)
- plot_jaccard_similarity(jaccard_df, comparison_type, word_cnt=100)
- jaccard_similarity(list1, list2)
- extract_year_week(year_week_str)

The idea was to analyze the sets of semantically related words, named as nearest neighbors, of
the target words “anfassen”, “angreifen” and “anlangen”. I wanted to compare the differences
between the both varieties, Austria and Germany, and to study the changes over time. The “Jaccard index”, 
which measures the similarity between finite sample sets, will serve as key metric. It can be calculated between:
- the word set of the first time period available and any given time period (per target word and variety) and 
- the word sets of the two varieties (per target word and time period).

The function `get_w2v_output()` loads all relevant `Word2Vec` models and returns the nearest neighbors of each target 
word in one list. This list is used as input in `calculate_jaccard_similarity()`, where a `pandas` DataFrame with all 
Jaccard similarity values is created. The values are calculated by using the helper function `jaccard_similarity()`. 
The function `plot_jaccard_similarity()` can generate two different plots, either a `years` or a `countries` comparison, 
depending on the argument `comparison_type`. The helper function `extract_year_week()` is used to extract the year and 
calendar week from the crawl name.

### How to run the program <hr>
If you want to collect and pre-process the data, as well as train a `Word2Vec` model,
you have to run the `train.py` file with these required parameters:
- `-c "crawl_name"`, e.g. `-c CC-MAIN-2024-38` (name of existing crawl)
- `-tld "top_level_domain"`, e.g. `-tld at` (existing top-level-domain)
- `-p "data_path"`, e.g `-p C:Documents` (main path for saving data folders)

Optionally, you can specify the number of `.wet` files, which should be downloaded from `Common Crawl`, with `-f`.
The default value is 500. If you want to change the `spaCy` model which is used for the lemmatisation part, 
you have to edit the variable `spacy_model`. The target words are hard-coded in the variable `target_words` 
and might also be changed for your analysis. There is always the option to run the file only for a specific 
subtask by putting `#` in front of the functions you do not want to execute.


If you want to evaluate and compare your trained models, you should run `evaluate.py`. 
There is just one required parameter (`-p "data_path"`, see above) to run the file,
but you should manually declare the following variables:
- `crawls`: vector of all crawls, e.g. `['CC-MAIN-2014-00', 'CC-MAIN-2019-35', 'CC-MAIN-2024-38']`
- `tlds`: vector of all top-level domains, e.g. `['at', 'de']`
- `target_words`: vector of all target words, e.g. `['angreifen', 'anfassen', 'anlangen']`
- `spacy_model`: string of name of `spaCy` model for data pre-processing, e.g. `'de_core_news_md'`
- `word_cnt`: integer of word set size of nearest neighbors of target words, e.g. `100`

### System information <hr>

- System: Windows
- Release: 10 
- Version: 10.0.20348 
- Machine: AMD64 
- Processor: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- Python version: 3.11.7
