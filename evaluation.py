# import packages
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
    return f'{year}-{week}'


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
        print(f"Load word2vec model for crawl '{year}' for top-level-domain '{tld}'")
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

    # sort years in right order
    years_sorted = sorted(years, key=lambda x: pd.to_datetime(x + '-1', format='%Y-%W-%w'))
    jaccard_df['year1'] = pd.Categorical(jaccard_df['year1'], categories=years_sorted, ordered=True)

    return jaccard_df


# function to plot the results
def plot_jaccard_similarity(jaccard_df, comparison_type, word_cnt=20):
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

        # map German words to shortcuts for visibility
        word_mapping = {
            'anfassen': 'w1',
            'angreifen': 'w2',
            'anlangen': 'w3'
        }
        df = df.copy()
        df['word1'] = df['word1'].map(word_mapping)
        df['word2'] = df['word2'].map(word_mapping)

        ax = sns.lineplot(
            x='year1', y='jaccard_idx', hue='country1', data=df,
            marker="o", palette="deep" #, jitter=True
        )
        # add text labels with word1 and word2 at each point
        max_j = max(df['jaccard_idx'])
        min_j = min(df['jaccard_idx'])
        for i in range(len(df)):
            year = df.iloc[i]['year1']
            jaccard_idx = df.iloc[i]['jaccard_idx']
            word1 = df.iloc[i]['word1']
            word2 = df.iloc[i]['word2']
            # check position and offset labels accordingly
            i += 1
            if i % 3 == 0:  # Alternate label placement to avoid overlap
                ax.text(year, jaccard_idx + 0.2 * max_j, f"{word1} vs {word2}",
                        fontsize=9, ha='right', rotation=25)
            elif i % 2 == 0:
                ax.text(year, jaccard_idx - 0.2 * max_j, f"{word1} vs {word2}",
                        fontsize=9, ha='right', rotation=25)
            else:
                ax.text(year, jaccard_idx, f"{word1} vs {word2}",
                        fontsize=9, ha='right', rotation=25)
        # specify axes
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(min_j - 0.25 * max_j, max_j + 0.25 * max_j)
        ax.set(xlabel='year', ylabel='jaccard similarity')
        ax.legend(title='country')
        plt.title(f'Jaccard Similarity with n = {word_cnt} ({comparison_type} comparisons)')

        # add word mapping legend
        text_str = "\n".join([f"{original} -> {new}" for original, new in word_mapping.items()])
        plt.gcf().text(x=0.5, y=0.7, s=text_str, fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))

    elif comparison_type == "countries":
        print("Compare word sets of different countries")    # for same target word and year
        df = jaccard_df[
            (jaccard_df['year1'] == jaccard_df['year2']) &
            (jaccard_df['country1'] != jaccard_df['country2']) &
            (jaccard_df['word1'] == jaccard_df['word2'])]
        #countries = df.country1 + df.country.unique()
        ax = sns.lineplot(
            x='year1', y='jaccard_idx', hue='word1', data=df, marker="o", palette="deep",
        )
        # specify axis labels
        ax.set(xlabel='year', ylabel='jaccard similarity')
        ax.legend(title='target words')
        plt.title(f'Jaccard Similarity with n = {word_cnt} ({comparison_type} comparisons)')  #{df[0, "country1"]} vs. {df[0, "country2"]}

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
        plt.title(f'Jaccard Similarity with n = {word_cnt} ({comparison_type} comparisons)')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # set arguments
    data_path = os.path.join("S:", "msommer")
    crawls = ['CC-MAIN-2014-00', 'CC-MAIN-2019-35', 'CC-MAIN-2024-38']
    tlds = ['at', 'de']
    target_words = ['angreifen', 'anfassen', 'anlangen']
    spacy_model = 'de_core_news_md'
    word_cnt = 100

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

    # plot for 'years' comparison
    plot_jaccard_similarity(jaccard_df, 'years', word_cnt)

    # plot for 'countries' comparison
    plot_jaccard_similarity(jaccard_df, 'countries', word_cnt)

    # plot for 'words' comparison
    plot_jaccard_similarity(jaccard_df, 'words', word_cnt)

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