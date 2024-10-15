########### imports ###########
import requests
from bs4 import BeautifulSoup

########### Global variables ###########
GENRE_URL = "https://blogger.ameba.jp/"


########### This file contains function meant for the most basic scraping
# before the initial abstract presentation ###########


def _scrape_topic_counts():
    """
    This function extracts the blog counts for each category in the Ameba
    blog overall category page
    :return: A dictionary in which every key is a topic and the value is the
    number of blogs associated with it.
    """
    url = GENRE_URL  # the website's URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    topic_elements = soup.find_all('li', class_='p-genreVerticalList__item')
    topics_dict = {}  # initialize the overall topics dict
    for element in topic_elements:
        main_title = element.find('h3',  # extract the main title
            class_='c-headingLv3 c-headingLv3--main').get_text(strip=True)
        # extract the subtitles and their values
        sub_titles = element.find_all('li',
            class_='p-genreVerticalList__subList__item')
        subtitle_dict = {}  # initialize the subtitle dict
        for subtitle in sub_titles:  # extract the subtitle
            subtitle_text = subtitle.find('span',
                class_='p-genreVerticalList__subList__genre').get_text(
                strip=True)
            value_text = (subtitle.find('span',  # extract the subtitle's counts
                class_='p-genreVerticalList__subList__users c-weakText').
                get_text(strip=True))
            # convert the count to integer and remove non-digit characters
            value = int(value_text.replace('人', '').replace(',', ''))
            # add pair to the subtitle dict
            subtitle_dict[subtitle_text] = value
        # store the main title and its subtitles in the dictionary
        topics_dict[main_title] = subtitle_dict
    return topics_dict


def extract_topic_counts():
    """
    This function sums all the sub-counts of each topic under each overall
    category (originally used to see which categories have the biggest traffic)
    :return: A sorted dictionary of the summed topics and categories
    """
    # get the topics dict from the website's data
    # structure is: {title1: {subtitle1: val1, subtitle2: val2...}, title2: ...}
    topics_dict = _scrape_topic_counts()
    topics_list = [topic for topic in topics_dict.keys()]  # use topic as keys
    # sum all subtitles' counts as values
    summed_topics_counts = [sum(sub_topic_dict.values()) for sub_topic_dict in
                            topics_dict.values()]
    # create a topic: overall counts dict
    topic_counts_dict = dict(zip(topics_list, summed_topics_counts))
    # sort the dict
    sorted_topic_counts_dict = dict(sorted(topic_counts_dict.items(),
                                           key=lambda item: item[1],
                                           reverse=True))
    return sorted_topic_counts_dict


########### Call space ###########
categories_dict = extract_topic_counts()
print(categories_dict)
