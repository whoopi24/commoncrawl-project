# 194.147 Interdisciplinary Project in Data Science <hr>
## An analysis of the development of the German touch verbs 'anfassen', 'angreifen', 'anlangen' with text data from `Common Crawl`.
### A comparison between Austria and Germany.
by Marina Sommer | 11778902 | TU Wien | October 2, 2024 <hr>


### Introduction <hr>
This project consists of the following files:
- train.py
- evaluation.py
- setup.py
- ReadMe.md

If you want to know how to run this project, go to the end of this file.
- aim of this project

### Data Collection <hr>
Included functions:
- get_files(crawl_dir, crawl_name, top_lvl_domain='at', files_cnt=500, skip=False)
- tryDownload(url, filename, retries=0)

### Data Preparation <hr>
Included functions:
- create_text_corpus(crawl_dir, top_lvl_domain='at', files_cnt=1000)
- preprocess_text_corpus_spacy(crawl_dir, spacy_model)
- count_pct_and_stopwords(text, stopwords)

### Model Training <hr>
Included functions:
- train_model(crawl_dir, spacy_model)

### Evaluation <hr>
Included functions:
- jaccard_similarity(list1, list2)
- get_w2v_output(data_path, crawl_names, top_lvl_domains, target_words, spacy_model, word_cnt=20)
- calculate_jaccard_similarity(word_sets)
- plot_jaccard_similarity(jaccard_df, comparison_type, word_cnt=20)
- extract_year_week(year_week_str)

### How to run the program <hr>
If you want to collect and pre-process the data, as well as train a word2vec model, 
you have to run the `train.py` file with these required parameters:
- `-c "crawl_name"`, e.g. `-c CC-MAIN-2024-38` (name of existing crawl)
- `-tld "top_level_domain"`, e.g. `-tld at` (existing top-level-domain)
- `-p "data_path"`, e.g `-p C:Documents` (main path for saving data folders)

Optionally, you can specify the number of `.wet` files, which should be downloaded from `Common Crawl`, with `-f`.
The default value is 500. If you want to change the spacy model which is used for the lemmatisation part, 
you have to edit the variable `spacy_model`. The target words are hard-coded in the variable `target_words` 
and might also be changed for your analysis. There is always the option to run the file only for a specific subtask/function.


If you want to evaluate and compare your trained models, you should run `evaluate.py`. 
There are no required parameters to run the file, but you should manually declare the following variables:
- `crawls`: vector of all crawls, e.g. `['CC-MAIN-2014-00', 'CC-MAIN-2019-35', 'CC-MAIN-2024-38']`
- `tlds`: vector of all top-level-domains, e.g. `['at', 'de']`
- `target_words`: vector of all target words, e.g. `['angreifen', 'anfassen', 'anlangen']`
- `spacy_model`: string of name of spacy model for data pre-processing, e.g. `'de_core_news_md'`
- `word_cnt`: integer of number of nearest neighbours of target words, e.g. `100`

### System information (example with timings)