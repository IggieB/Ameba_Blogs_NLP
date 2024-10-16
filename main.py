########### Imports ###########
import pandas as pd
import spacy
import ast
import nltk
import plotly.graph_objs as go
import MeCab
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
from transformers import (pipeline, AutoModelForSequenceClassification,
                          AutoTokenizer,)
from keybert import KeyBERT
import plotly.io as pio


########### Global Variables ###########
DATA_PATHS = {'fertility': 'Raw Blog Posts CSVs/fertility_20_w_text_clean.csv',
              'pregnancy journaling': 'Raw Blog Posts CSVs/pregnancy journaling_20_w_text_clean.csv',
              'parenting babies': 'Raw Blog Posts CSVs/parenting babies_20_w_text_clean.csv',
              'parenting toddlers': 'Raw Blog Posts CSVs/parenting toddlers_20_w_text_clean.csv',
              'parenting elementary+': 'Raw Blog Posts CSVs/parenting elementary+_20_w_text_clean.csv'}
TOKENIZED_PATHS = {'fertility': 'fertility_text_tokenized_2.csv',
              'pregnancy journaling': 'pregnancy '
                                      'journaling_text_tokenized_2.csv',
              'parenting babies': 'parenting babies_text_tokenized_2.csv',
              'parenting toddlers': 'parenting toddlers_text_tokenized_2.csv',
              'parenting elementary+': 'parenting '
                                       'elementary+_text_tokenized_2.csv'}
PLOT_COLORS = ['#5e4fa2', '#fee08b', '#3288bd', '#fdae61', '#66c2a5', '#f46d43',
              '#abdda4', '#d53e4f', '#e6f598', '#9e0142']
NLP = spacy.load('ja_ginza_electra')
# Additional stopwords added after examinations of the results
TOKENS_TO_REMOVE = ['ありがとう', 'ござる', 'ます', '年', '月', '日', '代', 'いただく',
                    '゚', '円', '楽天', '市場', 'だ', 'くださる', 'a', 'b', 'c',
                    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                    'z' 'って', 'じゃん', 'です', 'てる', 'しい','こそ', 'いく',
                    'こんな', 'とる', 'わ', 'つく', 'cm', 'ありがと', 'ユ', 'メ',
                    '才', '込', '某', 'https', 'ᴗ', '周', '田', 'け', 'キャ', 'けど',
                    'そんな', 'そよ', '▁', 'もち', 'たも', 'うい', 'ゆりな', 'ひかり',
                    'もりもり', 'きなこ', 'めろ', 'みみ', 'さや', 'かんな' 'ひめ',
                    'ねー', 'たい', 'よ', 'た', 'て', 'し', 'い', 'あ', 'えっ', '']


def tokenize_posts_ginza(posts_df, col_to_tokenize):
    """
    This function tokenizes the scraped posts using the GiNZA NLP Library
    (https://github.com/megagonlabs/ginza) and spaCy framework.
    :param posts_df:
    :param col_to_tokenize: Which column to tokenize (post text, post title,
    etc.)
    :return: The dataframe with the tokenized text in a new column.
    """
    text_to_tokenize = posts_df[col_to_tokenize]
    posts_links = posts_df['Post Link']
    tokenized_text = []
    for i in range(len(text_to_tokenize)):
        print(posts_links[i])  # for run follow up
        doc = NLP(str(text_to_tokenize[i]))
        # Initial stemming and stopwords removal
        doc_tokens = [token.lemma_ for token in doc if token.is_alpha and not
        token.is_stop]
        tokenized_text.append(doc_tokens)
    if col_to_tokenize == 'Post Text':
        posts_df['Tokenized Post Text Ginza'] = tokenized_text
        return posts_df
    elif col_to_tokenize == 'Post Title':
        posts_df['Tokenized Post Title Ginza'] = tokenized_text
        return posts_df


def import_stopwords():
    """
    THis function imports a Japanese stopwords file taken from the following
    github repository: https://github.com/stopwords-iso/stopwords-ja
    :return:
    """
    with open("stopwords_ja.txt", 'r', encoding='utf-8') as file:
        content = file.readlines()
        # Remove newline characters from each line and strip whitespace
        stopwords_list = [line.strip() for line in content]
    return stopwords_list


def stopwords_removal(tokenized_text):
    """
    Additional stopwords removal function (after examination of the initial
    results). This function uses the global list above (TOKENS_TO_REMOVE) to
    account for various changes which were done in the list during the
    analysis process.
    :param tokenized_text: The column of the text post-tokenization.
    :return: A list with the posts after stopwords removal.
    """
    stopwords_list = import_stopwords()
    clean_posts_list = []
    for post in tokenized_text:
        converted_post = ast.literal_eval(post)  # read the cell contents as
        # list and not as a string
        clean_post = [token for token in converted_post if token.lower() not in
                      TOKENS_TO_REMOVE and token not in stopwords_list and
                      token != 'って']  # 'って' required a literal reference
        clean_posts_list.append(clean_post)
        # for comparison and run follow up
        print(len(converted_post), len(clean_post))
    return clean_posts_list


def mecab_tokenizer(text):
    """
    An implementation of a tokenizer using MeCab as the tokenization model
    (https://github.com/SamuraiT/mecab-python3).
    :param text: The text (post / title) to tokenize
    :return: The tokenized text in the form of a list with the filtered tokens.
    """
    stopwords = import_stopwords()
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(text)  # Tokenize the text
    filtered_tokens = []
    while node:
        # Extract surface (the actual token) and feature
        # (POS and additional info)
        surface = node.surface
        feature = node.feature.split(',')
        # Check if the token is not a stopword or in TOKENS_TO_REMOVE
        if surface not in stopwords and surface not in TOKENS_TO_REMOVE:
            # Check if the token is one of the allowed character types
            pos = feature[0]
            # Only keep if it's a noun, verb, adjective, proper name, etc.
            if pos in ['名詞', '動詞', '形容詞', '固有名詞', '副詞', '接続詞']:
                # Further check the character type (optional)
                if all('\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF'
                        for char in surface):
                    filtered_tokens.append(surface)
        node = node.next
    return filtered_tokens


def plot_ngrams(tokenized_posts, category, n_values, top_n=10,
                color_scale='sunset'):
    """
    Thia function plots the most common n-grams in the tokenized posts' text
    processed earlier.
    :param tokenized_posts: The posts' text after tokenization
    :param category: Which category is currently being processed
    :param n_values: A list of integers indicating which n grams to
    produce (unigrams, bigrams, trigrams...)
    :param top_n: An integer indicating how many "slots" should be plotted
    for each type of n-gram.
    :param color_scale: Which color scale to use for the plots.
    :return: Nothing, generates the plots.
    """
    # Combine all tokenized posts into a single string
    documents = [' '.join(post) for post in tokenized_posts]
    for n in n_values:
        # Vectorize the documents to get n-grams
        vectorizer = CountVectorizer(ngram_range=(n, n))
        X = vectorizer.fit_transform(documents)
        # Sum the occurrences of each n-gram
        ngram_counts = X.sum(axis=0).A1
        ngram_frequencies = Counter(
            dict(zip(vectorizer.get_feature_names_out(), ngram_counts)))
        top_ngrams = ngram_frequencies.most_common(top_n)
        print(top_ngrams)
        # Separate the n-grams and their frequencies
        ngram_labels, frequencies = zip(*top_ngrams)
        # Create a horizontal Plotly bar plot with the color scale and add
        # frequency labels
        fig = go.Figure(data=[go.Bar(
            x=frequencies,
            y=ngram_labels,
            orientation='h',  # Horizontal bar plot
            marker=dict(color=frequencies, colorscale=color_scale),
            text=frequencies,  # Add frequency labels
            textposition='outside'  # Position labels outside the bars
        )])
        fig.update_layout(
            title=f"Top {top_n} {n}-grams",
            xaxis_title="Frequency",
            yaxis_title=f"{n}-grams",
            yaxis=dict(showgrid=True, automargin=True),
            font=dict(size=14))
        fig.show()
        pio.write_html(fig, file=f'{category}_top_{top_n}_{n}_grams.html',
                       auto_open=False)


def determine_optimal_clusters(tokenized_posts, max_clusters=200):
    """
    This function is designed to produce the ideal number of clusters for a
    given dataset (the tokenized posts) using the silhouette score method.
    :param tokenized_posts: The posts' text after tokenization
    :param max_clusters: The maximum number of clusters for which to
    calculate the silhouette score
    :return: The ideal number of clusters according to the calculation
    """
    # Convert the list of tokenized posts to strings for vectorization
    documents = [' '.join(post) for post in tokenized_posts]
    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(documents)
    # Initialize variables to store results
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    # Iterate through different numbers of clusters and calculate the
    # Silhouette score
    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        # Calculate the Silhouette score for the current number of clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(
            f"Number of clusters: {num_clusters}, Silhouette Score: {silhouette_avg}")
    # Plot the Silhouette scores for each number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.show()
    # Determine the optimal number of clusters
    optimal_clusters = cluster_range[
        silhouette_scores.index(max(silhouette_scores))]
    print(f"The optimal number of clusters is: {optimal_clusters}")
    return optimal_clusters


def cluster_and_generate_wordclouds(tokenized_posts, num_clusters,
                                    font_path=None):
    """
    This function clusters the dataset (tokenized posts), generates a scatter
    plot reflecting the overall division and then generates a wordcloud for
    each cluster.
    :param tokenized_posts: The posts' text after tokenization
    :param num_clusters: Hoe many clusters should be used
    :param font_path: Since the text in Japanese a font path was necessary.
    :return: Nothing generates wordcloud plots
    """
    # Convert the 2D list of tokens into a list of strings
    documents = [' '.join(post) for post in tokenized_posts]
    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(documents)
    # Apply K-Means clustering
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)
    # TSNE for reducing dimensions
    reducer = TSNE(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(X.toarray())
    # Scatter plot of the clustered data points
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=km.labels_,
                          cmap='viridis', alpha=0.7)
    plt.title('Clustered Data Points')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter)
    plt.show()
    # Generate and visualize word clouds for each cluster
    for i in range(num_clusters):
        # Extract documents that belong to the current cluster
        cluster_documents = [documents[j] for j in range(len(documents)) if km.labels_[j] == i]
        # Join all documents in the cluster into one string
        text = ' '.join(cluster_documents)
        # Generate the word cloud with a specified Japanese font
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Cluster {i}")
        plt.savefig(f'Cluster_{i}.png', format='png')
        plt.show()


def topic_modeling_bertopic(tokenized_posts):
    """
    This function executes topic modeling of the text using the BERTopic
    model developed by Maarten Grootendorst
    (https://maartengr.github.io/BERTopic/index.html).
    :param tokenized_posts: The posts' text after tokenization
    :return: Nothing, generates several plots of the results.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 5))
    documents = [' '.join(post) for post in tokenized_posts]
    topic_model = BERTopic(language="japanese",
                           calculate_probabilities=True,
                           verbose=True,
                           vectorizer_model=vectorizer)
    topics, probs = topic_model.fit_transform(documents)
    hierarchical_topics = topic_model.hierarchical_topics(documents)
    topic_info = topic_model.get_topic_info()
    print("Topic Information:\n", topic_info)
    # Saving two CSV files of the results
    topic_info.to_csv("Topic_Information.csv", encoding="utf-8-sig")
    doc_info = topic_model.get_document_info(documents)
    doc_info.to_csv("Document_Information.csv", encoding="utf-8-sig")
    # Visualizing results
    barchart = topic_model.visualize_barchart(top_n_topics=10,
                                              n_words=10)
    barchart.show()
    pio.write_html(barchart, file=f'Barchart plot.html', auto_open=False)
    hierarchy_chart = topic_model.visualize_hierarchy(
        hierarchical_topics=hierarchical_topics)
    hierarchy_chart.show()
    pio.write_html(hierarchy_chart, file=f'Hierarchy Chart plot.html',
                   auto_open=False)
    topics_chart_test = topic_model.visualize_topics()
    topics_chart_test.show()
    pio.write_html(topics_chart_test, file=f'Topics Chart plot.html',
                   auto_open=False)
    documents_chart = topic_model.visualize_documents(documents)
    documents_chart.show()
    pio.write_html(documents_chart, file=f'Documents Chart plot.html',
                   auto_open=False)


def sentiment_weighted_average(chunks_dict):
    """
    Due to size constraints this function take a dictionary of chunks of the
    original text (the full posts). The calculation is based on a
    combination of the chunk length, sentiment label (POSITIVE, NEGATIVE or
    NEUTRAL), and strength of the label score.
    :param chunks_dict: A dictionary of chunks (all a part of the same post).
    :return:The overall post sentiment and score
    """
    post_length = sum([info[0] for info in chunks_dict.values()])
    weighted_scores = {'POSITIVE': 0.0, 'NEUTRAL': 0.0, 'NEGATIVE': 0.0}
    for idx, (length, label, score) in chunks_dict.items():
        # Calculate the weight of the chunk based on its length relative to
        # the total post length
        weight = length / post_length
        # Add the weighted score to the corresponding sentiment
        weighted_scores[label] += weight * score
    final_sentiment = max(weighted_scores, key=weighted_scores.get)
    final_sentiment_score = weighted_scores[final_sentiment]
    return final_sentiment, final_sentiment_score


def sentiment_analysis(tokenized_posts, posts_df):
    """
    This function executes sentiment analysis for the posts using
    koheiduck/bert-japanese-finetuned-sentiment
    (https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)
    :param tokenized_posts: The posts' text after tokenization
    :param posts_df: The full dataframe stored in the CSV file
    :return: The classifications in a new column of the dataframe, saved into
    a new CSV file.
    """
    model_name = 'koheiduck/bert-japanese-finetuned-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model,
                           tokenizer=tokenizer)
    posts_sentiment, sentiments_score = [], []
    for i in range(len(tokenized_posts)):
        post = tokenized_posts[i]
        # diving the post to chunks due to sentiment model constraints
        chunks = [post[i:i + 200] for i in range(0, len(post), 200)]
        # key = index, val = [chuck length, chunk label, label score]
        chunks_dict = {}
        for j in range(len(chunks)):
            # classifying each chunk
            chunk_text = ' '.join(chunks[j])
            result = classifier(chunk_text)
            chunks_dict[j] = [len(chunks[j]), result[0]['label'], result[0][
                'score']]
            print(f"doc {i}, chunk {j}", result, chunks[j])
        # calculating the overall post sentiment if there is more than one chunk
        if len(chunks) > 1:
            final_sentiment, final_sentiment_score = (
                sentiment_weighted_average(chunks_dict))
        else:
            final_sentiment = chunks_dict[0][1]
            final_sentiment_score = chunks_dict[0][2]
        # Adding the current result to the previous ones
        posts_sentiment.append(final_sentiment)
        sentiments_score.append(final_sentiment_score)
        print(final_sentiment, final_sentiment_score)  # for run follow up
    # Adding data to the dataframe
    posts_df['Post Sentiment'] = posts_sentiment
    posts_df['Post Sentiment Score'] = sentiments_score
    return posts_df


def keyword_extraction(tokenized_posts, posts_df):
    """
    This function executes a Keyword extraction for each post in the
    dataframe using the KeyBERT model developed by Maarten Grootendorst
    (https://maartengr.github.io/KeyBERT/).
    :param tokenized_posts: The posts' text after tokenization
    :param posts_df: The full dataframe stored in the CSV file
    :return: The dataframe with the keywords added as a new column.
    """
    keywords_list = []
    # Using MeCab in this function produced better results
    vectorizer = CountVectorizer(tokenizer=mecab_tokenizer, ngram_range=(1, 3))
    kw_model = KeyBERT()
    for i in range(len(tokenized_posts)):
        result = kw_model.extract_keywords(tokenized_posts[i],
                                           vectorizer=vectorizer,
                                           keyphrase_ngram_range=(1, 3))
        print(i, result)
        keywords_list.append(result)
    posts_df['Post Keywords'] = keywords_list
    return posts_df


if __name__ == '__main__':
    ##### Tokenize Posts #####
    # for category, path in DATA_PATHS.items():
    #     print(category)
    #     category_df = pd.read_csv(path)
    #     tokenized_df = tokenize_posts_ginza(category_df, 'Post Text')
    #     tokenized_df.to_csv(f"{category}_text_tokenized.csv", encoding="utf-8-sig")
    ##### Tokenize Titles #####
    # for category, path in TOKENIZED_PATHS.items():
    #     print(category)
    #     category_df = pd.read_csv(path)
    #     tokenized_df = tokenize_posts_ginza(category_df, 'Post Title')
    #     tokenized_df.to_csv(f"{category}_text_tokenized_title.csv",
    #                         encoding="utf-8-sig")
    ##### Import CSVs #####
    tokenized_dfs = {}
    for category, path in TOKENIZED_PATHS.items():
        tokenized_dfs[category] = pd.read_csv(path)
    dfs_combined = pd.concat(tokenized_dfs.values(), axis=0, ignore_index=True)
    ##### Second round of stopwords removal #####
    all_clean_posts = stopwords_removal(dfs_combined['Tokenized Post Text Ginza'])
    # all_post_titles = stopwords_removal(dfs_combined['Tokenized Post Title Ginza'])
    clean_categories = {cat_name: stopwords_removal(cat_df['Tokenized Post Text Ginza'])
                        for cat_name, cat_df in tokenized_dfs.items()}
    ##### N-grams #####
    # for all data:
    # plot_ngrams(all_clean_posts, 'all', [1, 2, 3, 4, 5, 6, 7, 8], 15)
    # # per category:
    # for category, posts in clean_categories.items():
    #     print(f'N-grams in category: {category}')
    #     plot_ngrams(posts, category, [1, 2, 3, 4, 5, 6, 7, 8], 15)
    ##### Clustering #####
    # for all data:  # Running time impractical - maybe try again in the future
    # print(determine_optimal_clusters(all_clean_posts))
    # clusters_num = int(input("How many clusters are best for all: "))
    # cluster_and_generate_wordclouds(all_clean_posts,clusters_num,
    #                                 font_path='yumin.ttf')
    # per category:
    # for category, posts in clean_categories.items():
    #     print(f'Clustering in category: {category}')
    #     print(determine_optimal_clusters(posts))
    #     clusters_num = int(input("How many clusters are best for category: "))
    #     cluster_and_generate_wordclouds(posts, clusters_num,
    #                                     font_path='yumin.ttf')
    ##### Topic Modeling #####
    # for all data:
    # print(topic_modeling_bertopic(all_clean_posts))
    # per category:
    # for category, posts in clean_categories.items():
    #     print(f'Topic modelling in category: {category}')
    #     print(topic_modeling_bertopic(posts))
    #     continue_marker = input("Continue? ")
    ##### Keword Extraction ######
    # for category, path in TOKENIZED_PATHS.items():
    #     print(category)
    #     category_df = pd.read_csv(path)
    #     cat_keywords = keyword_extraction(category_df['Post Text'], category_df)
    #     cat_keywords.to_csv(f"{category}_keywords.csv", encoding="utf-8-sig")
    ##### Sentiment Analysis #####
    # for category, path in TOKENIZED_PATHS.items():
    #     print(category)
    #     category_df = pd.read_csv(path)
    #     sentiment_df = sentiment_analysis(clean_categories[category], category_df)
    #     sentiment_df.to_csv(f"{category}_sentiment.csv",
    #     encoding="utf-8-sig")




