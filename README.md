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
