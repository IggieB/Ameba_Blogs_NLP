########### Imports ###########
import requests
import time
import re
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

########### Global Variables ###########
CATEGORIES_DICT = {'fertility': 'ベビ待ち・不妊治療・妊活',
                   'pregnancy journaling': '妊娠記録',
                   'parenting babies': '子育て(ベビー)',
                   'parenting toddlers': '子育て(幼児)',
                   'parenting elementary+': '子育て(小学生以上)'}

CATEGORIES_RANKING_PAGE = \
    {'fertility': 'https://blogger.ameba.jp/genres/bebimachi/blogs/ranking',
     'pregnancy journaling':'https://blogger.ameba.jp/genres/maternity/blogs/ranking',
     'parenting babies': 'https://blogger.ameba.jp/genres/baby/blogs/ranking',
     'parenting toddlers': 'https://blogger.ameba.jp/genres/kids/blogs/ranking',
     'parenting elementary+': 'https://blogger.ameba.jp/genres/school-kids/blogs/ranking'}


GENERAL_USER_URL = "https://www.ameba.jp/profile/general/"
AMEBA_USER_URL = "https://www.ameba.jp/profile/ameba/"
USER_SHOP_URL = "https://ml.ameblo.jp/shops/"


def _get_blog_hashtags(url):
    """
    This function scrapes the hashtags most commonly used in each blog.
    :param url: The current ranking page from which the blogs are being scraped
    :return: The hashtags series as a list
    """
    driver = webdriver.Firefox()
    driver.get(url)
    wait = WebDriverWait(driver, 10)  # Wait for the elements to load
    ul_elements = wait.until(EC.presence_of_all_elements_located(
        (By.CLASS_NAME, 'c-hashtagList.u-clearfix')))
    blogs_hashtags = []  # Create a list to store all blogs hashtags
    for ul in ul_elements:  # Iterate through each <ul> element
        li_elements = ul.find_elements(By.CLASS_NAME, 'c-hashtagList__item')
        single_blog_hashtags = [li.text.strip() for li in li_elements]
        blogs_hashtags.append(single_blog_hashtags)
    driver.quit()  # close the driver
    return blogs_hashtags


def _extract_top_blog_tag(soup_object):
    """
    This function finds whether the blog has a top tag (monetized blog which
    requires a different url pattern later on).
    :param soup_object: The soup of the current page being proceed.
    :return: A list indicating which blogs have a top tag.
    """
    user_elements = soup_object.find_all('div', class_='p-rankingAllText__user')
    # Create a list to store whether each user has the "top blogger tag"
    has_top_blogger_tag = []
    for user_element in user_elements:
        # Check if the user element contains the "top blogger tag"
        top_blogger_tag = user_element.find('svg',
                                            class_='c-icon c-iconBlogger__icon')
        has_top_blogger_tag.append(top_blogger_tag is not None)
    return has_top_blogger_tag


def _extract_blogs_data(soup_object, url):
    """
    This function extracts the data of the blogs from the current ranking
    page being processed.
    :param soup_object: The soup of the current ranking  page being processed.
    :param url: The page url (necessary for the hashtag extraction)
    :return: All relevant details as lists.
    """
    main_div = soup_object.find('main', class_='l-contents__main')
    # find all blog elements: ranks, names, links, hashtags, usernames,
    # top blog tags and user page links.
    rank_tags = main_div.find_all('span', class_='c-iconRank__rank')
    blog_ranks = [rank_tag.text for rank_tag in rank_tags]
    blog_name_tags = main_div.find_all('h3',
                    class_='p-rankingAllText__title c-headingLv3 u-ellipsis')
    blog_names = [blog_name_tag.text for blog_name_tag in blog_name_tags]
    link_tags = main_div.find_all('a', attrs={'class': 'u-db'})
    blog_links = [a_tag['href'] for a_tag in link_tags]
    blog_links = [link for link in blog_links if '/entry-' not in link]
    # Find all common hashtags (has to use selenium for JS content)
    blogs_hashtags = _get_blog_hashtags(url)
    user_names = [link.split("/")[-2] for link in blog_links]
    top_blog_tags = _extract_top_blog_tag(main_div)
    user_page_links = []
    # Account for top tag in user page link creation
    for i in range(len(top_blog_tags)):
        if top_blog_tags[i]:
            user_page_link = AMEBA_USER_URL + user_names[i] + "/"
            user_page_links.append(user_page_link)
        else:
            user_page_link = GENERAL_USER_URL + user_names[i] + "/"
            user_page_links.append(user_page_link)
    return (blog_ranks, blog_names, blog_links, blogs_hashtags, user_names,
            top_blog_tags, user_page_links)


def _get_next_ranking_page(url, next_page_clicks):
    """
    This function scrapes the details of the blogs in each ranking page,
    going forward in the ranking list according to the number of clicks given to
    it.
    :param url: The url of the current category being processed.
    :param next_page_clicks: How many blogs should be scraped (intervals of 20)
    :return: All the relevant details of the blogs in the page as lists.
    """
    driver = webdriver.Firefox()  # Set up the firefox webdriver
    driver.get(url)  # Open the worldcat search page
    for i in range(next_page_clicks):  # get to 100 users
        next_prev_buttons = driver.find_elements(By.CLASS_NAME,
                                                 'c-pager__button')
        next_button = next_prev_buttons[0] if len(next_prev_buttons) == 1 \
            else next_prev_buttons[1]
        next_button.click()
        time.sleep(5)  # pause for dynamic page loading
    curr_ranking_page_html = driver.page_source
    soup = BeautifulSoup(curr_ranking_page_html, 'html.parser')
    (curr_page_ranks, curr_page_blog_names, curr_page_blog_links,
     curr_page_hashtags, curr_page_usernames, curr_page_top_tags,
     curr_page_user_pages) = _extract_blogs_data(soup, url)
    driver.quit()  # close the driver
    return (curr_page_ranks, curr_page_blog_names, curr_page_blog_links,
            curr_page_hashtags, curr_page_usernames, curr_page_top_tags,
            curr_page_user_pages)


def scrape_top_blogs(blogs_num, categories_dict, next_page_clicks):
    """

    :param blogs_num: How blogs should be scraped per category.
    :param categories_dict: The global categories dict for English-Japanese
    name conversion
    :param next_page_clicks: How many clicks are needed for the desired
    number of blogs.
    :return: A CSV file of all the scraped data.
    """
    for category, url in categories_dict.items():
        # Initializing all details' lists
        (blog_ranks, blog_names, blog_links, blogs_hashtags, blogs_usernames,
         top_tag_users, user_page_links) = [], [], [], [], [], [], []
        print(f" Working on {category} category")  # for run follow up
        for i in range(next_page_clicks):
            # get the same details of the current ranking page
            (page_ranks, page_blog_names, page_blog_links, page_hashtags,
             page_usernames, page_top_tags, page_user_pages) = (
                _get_next_ranking_page(url, i))
            # Append the results from the current page to the previous ones.
            blog_ranks += page_ranks
            blog_names += page_blog_names
            blog_links += page_blog_links
            blogs_hashtags += page_hashtags
            blogs_usernames += page_usernames
            top_tag_users += page_top_tags
            user_page_links += page_user_pages
        # Unite all category data to a dict and then a dataframe
        all_category_blogs_data = {'Category': [category] * blogs_num,
                              'Blog Rank': blog_ranks,
                              'Blog Name': blog_names,
                              'Blog Link': blog_links,
                              'Common Hashtags': blogs_hashtags,
                              'Username': blogs_usernames,
                              'Top User Tag': top_tag_users,
                              'User Page Link': user_page_links}
        category_df = pd.DataFrame(all_category_blogs_data)
        category_df.to_csv(f"{category}_blogs_ranking_data.csv",
                                 encoding="utf-8-sig")
    return "Dataframe generated"


def _scrape_user_job(soup_object):
    """
    This function finds the profession element in the user page and saves it.
    :param soup_object: The soup object of the current user page.
    :return: The field contents as a string.
    """
    job_element = soup_object.find('dl',
                            class_='user-info__list user-info__list--job')
    if job_element:  # if field has a value
        job_details = job_element.find_all('span',
                                           class_='user-info__sub-value')
        job_details_text = [detail.text for detail in job_details]
        return ", ".join(job_details_text)


def _scrape_user_details(soup_object):
    """
    This function is a general template for scraping various details from the
    user page (inserted in the original page with the same class)
    :param soup_object: soup_object: The soup object of the current user page.
    :return: A list of sets, each set includes a pair of the field name and
    it's value.
    """
    detail_elements = soup_object.find_all('dl', class_='user-info__list')
    user_details_pairs = []
    for element in detail_elements:
        detail_name = (element.find('dt', class_='user-info__term')).text
        detail_value = (element.find('dd', class_='user-info__value')).text
        user_details_pairs.append((detail_name, detail_value))
    return user_details_pairs


def scrape_user_data(blogs_df):
    """
    This function scrapes all details in each individual user page scraped
    earlier.
    :param blogs_df: The full blogs dataframe
    :return: The updated dataframe with the user's details.
    """
    user_page_links = blogs_df['User Page Link']
    all_users_details = {'Display Name': [], 'Status': [], 'Date of Birth': [],
                        'Gender': [], 'Blood Type': [], 'Originally From': [],
                        'Current Location': [], 'Occupation': []}
    category_to_column = {'性別': 'Gender', 'ステータス': 'Status',
                          '未既婚': 'Status', '生年月日': 'Date of Birth',
                          '血液型': 'Blood Type', '出身地': 'Originally From',
                          '居住地': 'Current Location', '職業': 'Occupation'}
    for i in range(len(user_page_links)):
        # Initializing the individual dictionary for the current user
        user_details = {'Display Name': '', 'Status': '', 'Date of Birth': '',
                        'Gender': '', 'Blood Type': '', 'Originally From': '',
                        'Current Location': '', 'Occupation': ''}
        response = requests.get(user_page_links[i])
        soup = BeautifulSoup(response.content, 'html.parser')
        display_name = (soup.find('h1', class_='user-info__name')).text
        user_details['Display Name'] = display_name
        user_details['Occupation'] = _scrape_user_job(soup)  # Find occupation
        user_details_tuples = _scrape_user_details(soup)  # Find other user details
        # Insert each detail to its corresponding user dict value
        for tuple in user_details_tuples:
            col_name = category_to_column[str(tuple[0])]
            user_details[col_name] = tuple[1]
        # Insert single user details to the overall dict
        for key, val in user_details.items():
            all_users_details[key].append(val)
        print(user_details)  # For run follow up
    users_details_df = pd.DataFrame.from_dict(all_users_details)
    blogs_users_df_united = pd.concat([blogs_df, users_details_df], axis=1,
                                      join='inner')
    return blogs_users_df_united


def _scrape_entry_details(entries_list):
    """
    This function scrapes the titles and page links of all posts within the
    entries list (generated in another function).
    :param entries_list: The list of posts to be scraped
    :return: A dictionary with two lists: one with the posts' titles and
    another with the posts' links.
    """
    all_entries_details = {'Post Title': [], 'Post Link': []}
    for entry in entries_list:
        title_element = entry.find('h2',  # Get post title
                                   {'data-uranus-component':'entryItemTitle'})
        if not title_element:  # Account for different page structure
            title_element = entry.find('a', href=True)
        entry_title = (title_element).text  # Get post title
        entry_link = title_element.get('href')  # Get post link
        if not entry_link:  # Account for different page structure
            a_tag = title_element.find('a')
            entry_link = a_tag.get('href')
        full_link = 'https://ameblo.jp' + entry_link
        print(entry_title, full_link)  # For run follow up
        all_entries_details['Post Title'].append(entry_title)
        all_entries_details['Post Link'].append(full_link)
    return all_entries_details


def scrape_blog_posts(blogs_users_df, category):
    """
    This function scrapes additional details regarding individual blog posts
    collected initially in previous runs
    :param blogs_users_df: The dataframe (CSV) in which details scraped
    earlier were saved.
    :param category: Which category the code is currently processing (
    different files).
    :return: A new dataframe with all the newly scraped details
    """
    all_posts_dict = {'Category': [], 'Username': [], 'Display Name': [],
                      'Post Title': [], 'Post Link': []}
    blog_links = blogs_users_df['Blog Link']
    usernames = blogs_users_df['Username']
    display_names = blogs_users_df['Display Name']
    for i in range(len(usernames)):
        curr_page_url = blog_links[i] + 'entrylist.html'
        print(curr_page_url)  # For running follow up
        response = requests.get(curr_page_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        post_list_elements = soup.find_all('li', class_='skin-borderQuiet')
        if len(post_list_elements) == 0:  # Accounting for a unique page template
            list_container = soup.find('ul',
                                       class_='contentsList skinBorderList')
            if list_container == None:  # Accounting for another page template
                list_container = soup.find('div', id='recent_entries_list')
            post_list_elements = list_container.find_all('li')
        single_user_posts = _scrape_entry_details(post_list_elements)
        # Add the newly scraped details to the previous ones
        all_posts_dict['Username'] += [usernames[i]] * len(
                        single_user_posts['Post Title'])
        all_posts_dict['Display Name'] += [display_names[i]] * len(
                        single_user_posts['Post Title'])
        all_posts_dict['Post Title'] += single_user_posts['Post Title']
        all_posts_dict['Post Link'] += single_user_posts['Post Link']
    # Insert values for the category column
    all_posts_dict['Category'] = [category] * len(all_posts_dict['Post Title'])
    blog_posts_df = pd.DataFrame.from_dict(all_posts_dict)  # Save data to a new df
    return blog_posts_df


def _scrape_single_post_time(post_soup):
    """
    This function scrapes the upload time of each post.
    :param post_soup: The post page soup object.
    :return: The upload time as a string.
    """
    # Find the script tag containing the time information
    script_tag = post_soup.find_all('script')[2].string.strip()
    time_data = re.findall(r'"datePublished":"(\d{4}-\d{2}-\d{2}T\d{2}:'
                           r'\d{2}:\d{2})', script_tag)[0]
    # Adjust time differences
    dt = datetime.strptime(time_data, "%Y-%m-%dT%H:%M:%S")
    adjusted_dt = dt - timedelta(hours=6)  # Subtract 6 hours
    adjusted_time_str = adjusted_dt.strftime("%Y-%m-%d")
    return adjusted_time_str


def _scrape_single_post_theme(post_soup):
    """
    This function scrapes the theme of each post (if inserted).
    :param post_soup: The post page soup object.
    :return: The post's theme as a string if exists, None otherwise.
    """
    script_tag = None
    script_tag_list = post_soup.find_all('script')
    for i in range(len(script_tag_list)):
        # Isolating the specific relevant element in the page
        if (script_tag_list[i].string and 'window.INIT_DATA' in
                script_tag_list[i].string):
            script_tag = script_tag_list[i].string.strip()
    theme_data = re.search(r'"theme_name":"(.*?)"', script_tag)
    if theme_data:
        return theme_data.group(1)
    else:
        return None


def _scrape_single_post_text(post_soup):
    """
    This function scrapes the text (contents) of each individual post.
    :param post_soup: The post page soup object.
    :return: The posts text after initial removal of irrelevant parts.
    """
    text_div = post_soup.find('div', {"id": "entryBody"})
    # Extract all text within the div element and its sub-elements
    blog_text = text_div.get_text(separator=' ', strip=True)
    cleaned_text = re.sub(r'\$\{[^}]*\}', '', blog_text)
    return cleaned_text


def _scrape_single_post_details(post_url):
    """
    This function scrapes the upload time, theme and text using previous
    sub-functions for each detail.
    :param post_url: The url os the current post being processed.
    :return: A dictionary with the relevant details.
    """
    single_post_data = {'Posting Date': '', 'Post Themes': '', 'Post Text': ''}
    # Accessing the post page
    response = requests.get(post_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    single_post_data['Posting Date'] = _scrape_single_post_time(soup)
    single_post_data['Post Themes'] = _scrape_single_post_theme(soup)
    single_post_data['Post Text'] = _scrape_single_post_text(soup)
    return single_post_data


def scrape_posts_data(posts_df):
    """
    This function goes over all posts in the dataframe and scrapes in-depth
    details (date, themes and text), saves into the dataframe and returns it.
    :param posts_df: A dataframe with initial details scraped earlier.
    :return: The dataframe updated with new details.
    """
    # Initializing a dictionary for storing the data
    all_posts_data = {'Posting Date': [], 'Post Themes': [], 'Post Text': []}
    posts_urls = posts_df['Post Link']
    for link in posts_urls:
        print(link)
        link_details = _scrape_single_post_details(link)
        # Add the new details to the dictionary
        all_posts_data['Posting Date'].append(link_details['Posting Date'])
        all_posts_data['Post Themes'].append(link_details['Post Themes'])
        all_posts_data['Post Text'].append(link_details['Post Text'])
    # Insert into the dataframe
    posts_df['Posting Date'] = all_posts_data['Posting Date']
    posts_df['Post Themes'] = all_posts_data['Post Themes']
    posts_df['Post Text'] = all_posts_data['Post Text']
    return posts_df


def find_empty_posts(posts_df):
    """
    Verification function for locating empty posts (arbitrarily chose a limit
    of less than 5 characters as the limit of "empty" posts).
    :param posts_df: The full posts dataframe with all the details scraped
    earlier.
    :return:The dataframe post removal of the empty posts.
    """
    posts_urls = posts_df['Post Link']
    posts_text = posts_df['Post Text']
    inds_to_drop = []
    for i in range(len(posts_urls)):
        post_chars = list(str(posts_text[i]))
        if len(post_chars) < 5:  # find posts with less than 5 characters
            print(posts_urls[i])
            inds_to_drop.append(i)
    clean_df = posts_df.drop(index=inds_to_drop)
    return clean_df



########### Call Space ###########
########### Initial Ranking Scraping ###########
# print(scrape_top_blogs(100, CATEGORIES_RANKING_PAGE, 5))
# 2.9.24 last run
########### Scraping Users Data ###########
# for category, path in file_paths.items():
#     print(category)
#     category_df = pd.read_csv(path)
#     user_details_df = scrape_user_data(category_df)
#     user_details_df.to_csv(f"{category}_user_details.csv", encoding="utf-8-sig")
# 2.9.24 last run
###### Scraping last 20 post titles ######
# file_paths = {'fertility': 'fertility_user_details.csv',
#               'pregnancy journaling': 'pregnancy journaling_user_details.csv',
#               'parenting babies': 'parenting babies_user_details.csv',
#               'parenting toddlers': 'parenting toddlers_user_details.csv',
#               'parenting elementary+': 'parenting elementary+_user_details.csv'}
# for category, path in file_paths.items():
#     print(category)
#     category_df = pd.read_csv(path)
#     category_init_posts = scrape_blog_posts(category_df, category)
#     category_init_posts.to_csv(f"{category}_last_20.csv", encoding="utf-8-sig")
# 2.9.24 last run
###### Scraping posts text ######
# file_paths = {'fertility': 'fertility_last_20.csv',
#               'pregnancy journaling': 'pregnancy journaling_last_20.csv',
#               'parenting babies': 'parenting babies_last_20.csv',
#               'parenting toddlers': 'parenting toddlers_last_20.csv',
#               'parenting elementary+': 'parenting elementary+_last_20.csv'}
# for category, path in file_paths.items():
#     print(category)
#     category_df = pd.read_csv(path)
#     category_posts_w_text = scrape_posts_data(category_df)
#     category_posts_w_text.to_csv(f"{category}_20_w_text.csv",
#                                  encoding="utf-8-sig")
###### Find problematic posts ######
# file_paths = {'fertility': 'fertility_20_w_text.csv',
#               'pregnancy journaling': 'pregnancy journaling_20_w_text.csv',
#               'parenting babies': 'parenting babies_20_w_text.csv',
#               'parenting toddlers': 'parenting toddlers_20_w_text.csv',
#               'parenting elementary+': 'parenting elementary+_20_w_text.csv'}
# for category, path in file_paths.items():
#     print(category)
#     category_df = pd.read_csv(path)
#     clean_category_df = find_empty_posts(category_df)
#     clean_category_df.to_csv(f"{category}_20_w_text_clean.csv",
#                                  encoding="utf-8-sig")