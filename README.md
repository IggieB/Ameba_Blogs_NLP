<div align="center">
  <h1 align="center">Blogs NLP Analysis - Japan's Birth Crisis Research</h1>
  <h3>NLP Project for Funsies</h3>

<a><img src="https://images.unsplash.com/photo-1597590094308-e283f0f2030b?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2FuZHxlbnwwfHwwfHx8MA%3D%3D" alt="Photo by Kenny Gaines on Unsplash" style="width:250px;height:375px"></a>

</div>

<br/>

## General Description ✨

This project was an initial attempt to use pre-made NLP models such as RoBERTa, DistilRoBERTa, and others to perform sentiment and emotion analysis on text portions 
in Harry Potter that mention spells explicitly. The intention was to produce certain insights regarding the representation of magic (and specifically spells) using 
the results. 

## Project Steps 🪄

- Scraping all spells appearing in the series to produce a comprehensive list
- Finding all sentences (and later paragraphs) in the books in which those spells appear
- Processing all the text samples for the analysis
- Running the actual sentiment and emotion analysis using pre-made models
- Plotting the results after saving them as CSV files

#### 1. Spells Scraping - "spells_scraping.py" ✨

All spells were scraped from [Harry Potter Wiki list of spells](https://harrypotter.fandom.com/wiki/List_of_spells) using BeautifulSoup and saved as 
a regular txt file. Out of the said list, 2 spells had to be removed and later added manually ("Pack" and "Point Me") as their phrasing is too similar to everyday language and
they produced a lot of false positives in later stages.

#### 2 + 3. Spells Tracing and Processing - "find_spells_sentences.py" 🪄

The text sampling was done by cross-referencing each sentence in the book (post-tokenization) with a spell list saved in the previous step, and saving the text whenever there's
an intersection. Each sample was saved in 2 forms - the sentence itself ("all_books_spells_sentences.csv") and the surrounding paragraph containing 2 additional sentences before
and after the target sentence for extra context ("all_books_spells_paragraphs.csv"). The samples were later processed for analysis, including the removal of stopwords, 
punctuation, and lemmatization. Finally, the specific sentences containing the problematic spells mentioned earlier were added and processed using separate functions.

#### 4. Sentiment and Emotion Analysis - "main.py" ✨

Due to time constraints (and interest in testing the usability of pre-made models), all models used in this project were run without additional training. As context impacted the
results, sentiment analysis was done on both sentence and paragraph samples, whereas emotion analysis was done only for paragraph samples. The models used for sentiment analysis are:

- [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [NLTK VADER](https://www.nltk.org/howto/sentiment.html)

And those for emotion analysis are:
- [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- [emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

Finally, the results were all saved as CSV files.


#### 5. Plotting The Results - "plots.py" 🪄

The analysis results were plotted using Seaborn and Matplotlib. The focus was mostly on the presence of different sentiments/emotions throughout the books hoping to get a certain
"timeline" for changes if there are any, and which spells were often categorized as specific sentiments/emotions, to check whether there are certain underlying connotations for
certain spells (and if so, which).

```

## Disclaimer

All rights regarding the raw data used in this project (The Harry Potter books, Wizarding World and all
associated concepts) belong to J.K.Rowling. They were used as experimenting material only, and for no monetary
gains of any kind.
