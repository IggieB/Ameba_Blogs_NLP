<div align="center">
  <h1 align="center">Blogs NLP Analysis - Japan's Birth Crisis Research</h1>
  <h3>Another NLP Project for Fun</h3>

<a><img src="https://www3.nhk.or.jp/news/html/20240605/K10014471471_2406051701_0605180037_02_03.jpg" alt="Photo published in the article 「去年の合計特殊出生率 1.20で過去最低に 東京は「1」を下回る」on the news site NHK" style="width:640px;height:360px"></a>

</div>

<br/>

## General Description 📋

This project is part of a seminar paper attempting to explore which insights regarding Japan's birth crisis can be found in related blogs on the popular platform [Ameba Blog](https://www.ameba.jp/). 10,000 posts were sampled based on the internal ranking lists of five related categories (fertility, pregnancy journaling, childrearing, etc.) and processed using several transformer-based models for Japanese corpora. 

## Project Steps 📝

- Scraping the 100 leading blogs (and user details) in the five categories chosen for the project (2.9.24)
- Scraping the last 20 posts from each blog (2.9.24)
- Pre-processing all the text samples for the analysis
- Processing the posts, which included n-gram generation, clustering, topic modeling and sentiment analysis (a keyword extraction function was also written but the results were not used)
- Plotting the results after saving them as CSV files

#### 1. Blog Scraping - "blog_scraping.py" 📋

The leading 100 leading blogs (and user details) were scraped from the five following categories in Ameba Blog: [ベビ待ち・不妊治療・妊活](https://blogger.ameba.jp/genres/bebimachi), [妊娠記録](https://blogger.ameba.jp/genres/maternity), [子育て(ベビー)](https://blogger.ameba.jp/genres/baby), [子育て(幼児)](https://blogger.ameba.jp/genres/kids) and [子育て(小学生以上)](https://blogger.ameba.jp/genres/school-kids). Afterward, the last 20 posts were scraped from each blog and all information was saved as CSV files (available in User Ranking CSVs folder and Raw Blog Posts CSVs). It is important to note that posts with less than 5 characters (containing visual content only) were removed, after which 9836 posts remained as a sample.

#### 2 + 3. Posts Pre-Processing and Processing - "main.py" 📝

The text sampling was done by cross-referencing each sentence in the book (post-tokenization) with a spell list saved in the previous step, and saving the text whenever there's
an intersection. Each sample was saved in 2 forms - the sentence itself ("all_books_spells_sentences.csv") and the surrounding paragraph containing 2 additional sentences before
and after the target sentence for extra context ("all_books_spells_paragraphs.csv"). The samples were later processed for analysis, including the removal of stopwords, 
punctuation, and lemmatization. Finally, the specific sentences containing the problematic spells mentioned earlier were added and processed using separate functions.

#### 4. Sentiment and Emotion Analysis - "main.py" 📋

Due to time constraints (and interest in testing the usability of pre-made models), all models used in this project were run without additional training. As context impacted the
results, sentiment analysis was done on both sentence and paragraph samples, whereas emotion analysis was done only for paragraph samples. The models used for sentiment analysis are:

- [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [NLTK VADER](https://www.nltk.org/howto/sentiment.html)

And those for emotion analysis are:
- [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- [emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

Finally, the results were all saved as CSV files.


#### 5. Plotting The Results - "plots.py" 🪄

All rights regarding the raw data used in this project (the blog posts analyzed) belong to their original authors. They were used as research material only, and for no monetary gains of any kind.

```

## Disclaimer

All rights regarding the raw data used in this project (The Harry Potter books, Wizarding World and all
associated concepts) belong to J.K.Rowling. They were used as experimenting material only, and for no monetary
gains of any kind.
