########### Imports ###########
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from matplotlib.font_manager import FontProperties
import plotly.io as pio

########### Global Variables ###########
COLORS = ['#5e4fa2', '#fee08b', '#3288bd', '#fdae61', '#66c2a5', '#f46d43',
              '#abdda4', '#d53e4f', '#e6f598', '#9e0142']

USERS_FILES = {'fertility': 'User Ranking CSVs/fertility_user_details.csv',
              'pregnancy journaling': 'User Ranking CSVs/pregnancy journaling_user_details.csv',
              'parenting babies': 'User Ranking CSVs/parenting babies_user_details.csv',
              'parenting toddlers': 'User Ranking CSVs/parenting toddlers_user_details.csv',
              'parenting elementary+': 'User Ranking CSVs/parenting elementary+_user_details.csv'}

POSTS_FILES = {'fertility': 'fertility_sentiment.csv',
              'pregnancy journaling': 'pregnancy journaling_sentiment.csv',
              'parenting babies': 'parenting babies_sentiment.csv',
              'parenting toddlers': 'parenting toddlers_sentiment.csv',
              'parenting elementary+': 'parenting elementary+_sentiment.csv'}


def unite_dataframes(df_dict):
    """
    This function is used to unite all separate CSV files (of the users or
    the posts) to a single data frame for additional processing.
    :param df_dict: A dataframe dictionary of either the user data files or
    the blog posts data files
    :return: The united dataframe
    """
    all_dfs_list = []
    for key, val in df_dict.items():
        cat_df = pd.read_csv(val)  # Read each file as a dataframe
        all_dfs_list.append(cat_df)
    combined_df = pd.concat(all_dfs_list, ignore_index=True)
    return combined_df


def user_details_pie_chart(data_df, data_col, font_path=None, label_threshold=2):
    """
    This function generate pie charts for several fields regarding the users'
    profiles.
    :param data_df: The CSV file data loaded as a dataframe
    :param data_col: Which column (details) should be plotted.
    :param font_path: Custom font for Japanese text in the plot
    :param label_threshold: The minimum value that should appear explicitly
    in the chart.
    :return: Nothing, generates the plots.
    """
    title_dict = {'Status': 'Marriage Status Distribution',
                  'Gender': 'Gender Distribution',
                  'Blood Type': 'Blood Type Distribution',
                  'Originally From': 'Origin Location Distribution',
                  'Current Location': 'Current Location Distribution',
                  'Occupation': 'Occupation Distribution',
                  'Date of Birth': 'Age Distribution',
                  'Top User Tag': 'Top Users (Top Tag Distribution)'}
    # Adjust the list for unknown or long entries
    data_list = ['Unknown' if isinstance(line, float) else line for line in
                 data_df[data_col]]
    data_counts = Counter(data_list)  # Count the occurrences of each value
    labels = list(data_counts.keys())
    values = list(data_counts.values())
    # Sort the labels and values based on the count in descending order
    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*sorted_pairs)
    custom_font = FontProperties(fname=font_path)  # Load custom font
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    # Calculate percentages
    total = sum(values)
    percentages = [(v / total) * 100 for v in values]
    # Function to place text inside or outside based on percentage
    def autopct_fn(pct):
        return f'{pct:.1f}%' if pct > label_threshold else ''
    # Plot the pie chart
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct=autopct_fn,
                                      colors=COLORS,
                                      startangle=90, wedgeprops=dict(width=0.3),
                                      pctdistance=0.8)
    # Set font for labels inside the pie
    for text, autotext, pct in zip(texts, autotexts, percentages):
        text.set_fontproperties(custom_font)
        autotext.set_fontproperties(custom_font)
        # Remove text inside the pie for smaller slices based on a threshold
        if pct < label_threshold:
            text.set_text("")  # Remove the label if below the threshold
    # Add counts to the labels in the legend
    legend_labels = [f"{label}: {count}" for label, count in
                     zip(labels, values)]
    # Add a legend with the updated labels and colors
    ax.legend(wedges, legend_labels,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), prop=custom_font, fontsize=12)
    # Set the custom font for the title
    ax.set_title(title_dict[data_col], fontproperties=custom_font, fontsize=20)
    # Ensure layout is tight so that everything fits
    plt.tight_layout()
    plt.show()


def sentiment_pie_chart(data_df, data_title, font_path=None, min_pie_width=8):
    """
    This function generates a pie chart showing the distribution of the
    sentiment analysis done earlier.
    :param data_df: The dataframe to be used for the plot
    :param data_title: A string that will be added to the plot title for
    clarifying which data was used
    :param font_path: A custom font path for text in Japanese
    :param min_pie_width: A minimum width for the pie itself
    :return: Nothing, generates the plot
    """
    # Count the occurrences of each sentiment
    sentiments_list = list(data_df['Post Sentiment'])
    sentiment_counts = Counter(sentiments_list)
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    # Sort the labels and values based on the count in descending order
    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*sorted_pairs)
    # Load custom font if provided
    custom_font = FontProperties(fname=font_path)
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(min_pie_width, 8))
    # Create the pie chart with custom font and remove label threshold
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                      textprops={'fontproperties': custom_font},
                                      colors=COLORS)
    # Add a legend with sorted labels, counts, and custom font
    legend_labels = [f"{label}: {count}" for label, count in
                     zip(labels, values)]
    ax.legend(wedges, legend_labels,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), prop=custom_font)
    # Set the title with the custom font (if provided)
    plot_title = data_title + " Sentiment Distribution"
    ax.set_title(plot_title, fontproperties=custom_font,
                 fontsize=16)
    plt.tight_layout()
    plt.show()


def flatten_keywords(data_df, col_name):
    """
    This function flattens the keywords column for topic modeling of the
    keywords themselves.
    :param data_df: The full dataframe
    :param col_name: The column name (for possible future re-use of the
    function)
    :return: The flattened keywords
    """
    all_flattened_lines = []
    for line in data_df[col_name]:
        line = ast.literal_eval(line)  # read value as a data type and not str
        flat_line = [key_tuple[0] for key_tuple in line]
        all_flattened_lines.append(flat_line)
    return all_flattened_lines


def generate_themes_ngrams(themes_col, category, top_n=10,
                           color_scale='sunset'):
    """
    This function generates a horizontal bar plot mapping the most common
    themes appearing in posts (as tagged by the users themselves).
    :param themes_col: The themes column from the dataframe
    :param category: Which category is currently being plotted
    :param top_n: How many values should be plotted
    :param color_scale: Which color scale to use
    :return: Nothing, generates the plot and saves it as an HTML file
    """
    counter_dict = Counter(themes_col)
    sorted_counter_dict = dict(
        sorted(counter_dict.items(), key=lambda x: x[1], reverse=True))
    top_themes = list(sorted_counter_dict.keys())[:top_n]
    top_frequencies = list(sorted_counter_dict.values())[:top_n]
    # Create a horizontal bar plot using Plotly
    fig = go.Figure(go.Bar(
        x=top_frequencies,
        y=top_themes,
        orientation='h',  # horizontal bar plot
        marker=dict(
            colorscale=color_scale,  # Use the specified color scale
            color=top_frequencies,  # Color the bars according to frequency
        ),
        text=top_frequencies,  # Show frequency values on the bars
        textposition='auto'  # Automatically position text
    ))
    # Customize the layout
    fig.update_layout(
        title=f"Top Themes for {category}",
        xaxis_title="Frequency",
        yaxis_title="Themes",
        yaxis={'categoryorder': 'total ascending'},
        # Sort y-axis based on values
        template='plotly',
    )
    fig.show()
    pio.write_html(fig, file=f'{category}_top_{top_n}_themes.html',
                   auto_open=False)


########### Call Space ###########
all_users_df = unite_dataframes(USERS_FILES)  # unite all user categories dfs
all_posts_df = unite_dataframes(POSTS_FILES)  # unite all posts categories dfs
##### Users Info Pie Charts #####
# pie_chart_option = ['Status', 'Gender', 'Blood Type', 'Originally From',
#                     'Current Location', 'Occupation', 'Date of Birth',
#                     'Top User Tag']
# for val in pie_chart_option:
#     user_details_pie_chart(all_users_df, val, 'yumin.ttf')
##### Sentiments Pie Charts #####
# for all data:
# sentiment_pie_chart(all_posts_df, "All Posts", 'yumin.ttf')
# per category:
# for category, path in POSTS_FILES.items():
#     category_df = pd.read_csv(path)
#     sentiment_pie_chart(category_df, category.title(), 'yumin.ttf')
##### Keywords Topic Modeling #####
# flattened_keywords_col = flatten_keywords(all_posts_df, 'Post Keywords')
# topic_modeling_bertopic(flattened_keywords_col)
##### Themes N-grams #####
# For all data:
# generate_themes_ngrams(all_posts_df['Post Themes'], 'All Posts', 15)
# per category:
# for category, path in POSTS_FILES.items():
#     category_df = pd.read_csv(path)
#     generate_themes_ngrams(category_df['Post Themes'], category.title(), 15)


