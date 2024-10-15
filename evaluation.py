# import packages
import argparse
import os
import spacy
from gensim.models import Word2Vec
from itertools import product, combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# download spacy german language models in terminal
# python -m spacy download de_core_news_sm
# python -m spacy download de_core_news_md
# python -m spacy download de_core_news_lg


# function to extract the year and week from the string
def extract_year_week(year_week_str):
    # Split the string by '-' and extract the relevant parts
    year_week = year_week_str.split('-')
    year = int(year_week[2])
    week = int(year_week[3])
    if week < 10:
        week = str(week).rjust(2, '0')
    return f'{year}-{week}'


# function to calculate the jaccard index of two lists or sets
def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)  # set conversion for list input
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


# function to get the word sets with the nearest neighbours of the word2vec models
def get_w2v_output(data_path, crawl_names, top_lvl_domains, target_words, spacy_model, word_cnt=20):
    year_tld = list(product(crawl_names, top_lvl_domains))
    word_lists = {}
    for year, tld in year_tld:
        print(f"Load word2vec model for crawl '{year}' for top-level domain '{tld}'")
        # load model
        model_fname = os.path.join(data_path, year, tld, "word2vec_" + spacy_model + ".model")
        model = Word2Vec.load(model_fname)

        # get nearest neighbours of target words
        for word in target_words:
            key = tuple([extract_year_week(year), tld, word])
            # https://stackoverflow.com/questions/50275623/difference-between-most-similar-and-similar-by-vector-in-gensim-word2vec
            word_lists[key] = [result[0] for result in model.wv.similar_by_word(word, word_cnt)]

    return word_lists


# function to calculate the jaccard index for the specific word sets
def calculate_jaccard_similarity(word_sets):
    years = set([key[0] for key in word_sets.keys()])
    countries = set([key[1] for key in word_sets.keys()])
    words = set([key[2] for key in word_sets.keys()])
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
                        jaccard_list.append([year, year, country, country, word1, word2, similarity])

    # create dataframe with all comparisons
    jaccard_df = pd.DataFrame(jaccard_list, columns=[
        'year1', 'year2', 'country1', 'country2', 'word1', 'word2', 'jaccard_idx'
    ])

    return jaccard_df


# function to plot the results
def plot_jaccard_similarity(jaccard_df, comparison_type, word_cnt=20):
    # plot figure
    plt.figure(figsize=(10, 6))

    # check type of comparison
    type_vec = ['countries', 'years']
    if comparison_type not in type_vec:
        raise ValueError(f"'{type}' is not correct! Choose between {type_vec}!")
    elif comparison_type == "countries":
        print("Compare word sets of different countries")    # for same target word and year
        df = jaccard_df[
            (jaccard_df['year1'] == jaccard_df['year2']) &
            (jaccard_df['country1'] != jaccard_df['country2']) &
            (jaccard_df['word1'] == jaccard_df['word2'])]
        ax = sns.lineplot(
            x='year1', y='jaccard_idx', hue='word1', data=df, marker="o", palette="deep",
        )
        # specify axis labels
        ax.set(xlabel='year', ylabel='jaccard similarity')
        ax.legend(title='target words')
        plt.title(f"Jaccard Similarity with n = {word_cnt} ({df['country1'].iloc[0]} vs. {df['country2'].iloc[0]})")  # {comparison_type} comparisons

    elif comparison_type == "years":
        print("Compare word sets of different years")        # for same target word and country
        df = jaccard_df[
            (jaccard_df['year1'] != jaccard_df['year2']) &
            (jaccard_df['country1'] == jaccard_df['country2']) &
            (jaccard_df['word1'] == jaccard_df['word2'])]
        ax = sns.lineplot(
            x='word1', y='jaccard_idx', hue='country1', data=df, marker="o", palette="deep",
        )
        # specify axis labels
        ax.set(xlabel='target words', ylabel='jaccard similarity')
        ax.legend(title='country')
        plt.title(f"Jaccard Similarity with n = {word_cnt} ({df['year1'].iloc[0]} vs. {df['year2'].iloc[0]})")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# function for using arguments when running the file
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="specifies data path"
    )
    return parser.parse_args()


if __name__ == '__main__':

    # get data path argument
    args = get_args()
    data_path = args.path

    # set arguments
    crawls = ['CC-MAIN-2014-00', 'CC-MAIN-2019-35', 'CC-MAIN-2024-38']
    tlds = ['at', 'de']
    target_words = ['anfassen', 'angreifen', 'anlangen']
    spacy_model = 'de_core_news_md'
    word_cnt = 100

    # check number of selected TLDs
    if len(tlds) > 2:
        raise ValueError(f"Please provide at most two top-level domains for comparison!")

    # check target words
    nlp = spacy.load(spacy_model)
    vocab = set(nlp.vocab.strings)
    for word in target_words:
        if word not in vocab:
            raise ValueError(f"'{word}' is not in spacy '{spacy_model}' vocabulary!")

    # get word sets
    output = get_w2v_output(data_path, crawl_names=crawls, top_lvl_domains=tlds,
                            target_words=target_words, spacy_model=spacy_model, word_cnt=word_cnt)

    # calculate jaccard index
    jaccard_df = calculate_jaccard_similarity(output)

    # plot for 'countries' comparison
    # sort words and crawls in right order
    crawls = [crawl[-7:] for crawl in crawls]
    jaccard_df['word1'] = pd.Categorical(jaccard_df['word1'], categories=target_words, ordered=True)
    jaccard_df['year1'] = pd.Categorical(jaccard_df['year1'], categories=crawls, ordered=True)
    plot_jaccard_similarity(jaccard_df, 'countries', word_cnt)

    # plots for 'years' comparison
    # sort words and tlds in right order
    jaccard_df['word1'] = pd.Categorical(jaccard_df['word1'], categories=target_words, ordered=True)
    jaccard_df['country1'] = pd.Categorical(jaccard_df['country1'], categories=tlds, ordered=True)
    n = len(crawls)
    if n > 2:
        target_year = crawls[0]
        jaccard_df = jaccard_df[(jaccard_df['year1'] == target_year) |
                                (jaccard_df['year2'] == target_year)]
        for i in range(1, n):
            target_year = crawls[i]
            plot_df = jaccard_df[(jaccard_df['year1'] == target_year) |
                                 (jaccard_df['year2'] == target_year)]
            plot_jaccard_similarity(plot_df, 'years', word_cnt)
    else:
        plot_jaccard_similarity(jaccard_df, 'years', word_cnt)
