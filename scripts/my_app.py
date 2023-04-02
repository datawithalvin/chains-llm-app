import pandas as pd
import snscrape.modules.twitter as sntwitter
from snscrape.modules.twitter import TwitterTweetScraperMode
import streamlit as st
import re

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter
from langchain.docstore.document import Document

import os
from dotenv import load_dotenv

import nltk
nltk.download("punkt")

load_dotenv()


def scrape_twitter_thread(url):
    """
    Scrape a Twitter thread from a given URL and return the thread content and engagement statistics.

    Args:
    url (str): The URL of the Twitter thread to scrape.

    Returns:
    str: The content of the Twitter thread, along with the number of views, likes, retweets, and quotes.

    Raises:
    ValueError: If the URL is invalid.

    """

    # Extract the username and tweet ID from the URL using regex
    match = re.match(r'https?://(?:www\.|mobile\.)?twitter\.com/\w+/status/(\d+)', url)
    if match is None:
        raise ValueError("Invalid Twitter thread URL")
    username = url.split('/')[3]
    start_thread = int(match.group(1))

    # Scrape the tweets in the conversation thread
    tweets = sntwitter.TwitterTweetScraper(start_thread, mode=TwitterTweetScraperMode.SCROLL).get_items()

    # Create a list to store tweet data
    tweets_data = []

    # Iterate through the tweets and store relevant data in the list
    for tweet in tweets:
        # Clean the tweet content by replacing newlines, URLs, and hashtags with spaces
        content = tweet.rawContent.replace('\n', ' ').replace('http\S+', ' ').replace('#\S+', ' ')
        tweets_data.append([content, tweet.user.username, tweet.likeCount, tweet.viewCount, tweet.retweetCount, tweet.quoteCount])

    # Create a DataFrame from the list
    columns = ['Content', 'User', 'Likes', 'Views', 'Retweets', 'Quotes']
    df = pd.DataFrame(tweets_data, columns=columns)

    # Filter the DataFrame to include only tweets from the thread owner
    filtered_df = df[df['User'] == username]

    # Combine the content of all tweets into a single text file
    content = '\n\n'.join(filtered_df['Content'].tolist())

    # Get information about the thread from the first row
    views = filtered_df['Views'].iloc[0]
    likes = filtered_df['Likes'].iloc[0]
    retweets = filtered_df['Retweets'].iloc[0]
    quotes = filtered_df['Quotes'].iloc[0]

    # Return the content and engagement statistics as a string
    output_str = f"{content}\n\nThis thread was viewed by {views} account(s), liked by {likes} account(s), retweeted by {retweets} account(s), and quoted by {quotes} account(s)."
    return output_str


def get_thread_summary(output_thread):
    """
    This function takes in a Twitter thread and returns a concise summary of the thread with casual tone
    and displays the viewed counts, likes counts, retweets counts, and quotes counts at the end of the summary 
    all in Bahasa Indonesia.

    Args:
    output_thread (str): A Twitter thread as a string.

    Returns:
    str: A concise summary of the thread with the specified tone and counts.

    """
    # Split the Twitter thread into individual documents
    text_splitter = NLTKTextSplitter(separator="\n")
    texts = text_splitter.split_text(output_thread)
    threads = [Document(page_content=t) for t in texts[:3]]

    # Set the prompt template for the summarization task
    prompt_template = """ Write a concise summary of the following twitter thread:

    {text}

    CONCISE THREAD SUMMARY WITH CASUAL TONE AND DISPLAY THE VIEWED COUNTS, 
    LIKES COUNTS, RETWEETS COUNTS, AND QUOTES COUNTS AT THE END OF THE SUMMARY ALL IN BAHASA INDONESIA":
    """

    # Initialize the summarization chain and run the chain on the Twitter thread
    api_key = os.environ["OPENAI_API_KEY"]
    myprompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(OpenAI(temperature=0.6, 
                                        openai_api_key=api_key, 
                                        model_name="gpt-3.5-turbo",
                                        max_tokens=256), 
                                chain_type="stuff", prompt=myprompt)

    return chain.run(threads)

# Set the page configuration and title
st.set_page_config(page_title="Ringkas Thread Twitter", page_icon=":memo:")
st.title("Ringkas Thread Twitter")

# Display a brief description of the web app in Bahasa Indonesia with casual tone
st.markdown("""
    Aplikasi web ini membantu merangkum thread Twitter dengan mengekstrak konten thread dan menghasilkan ringkasan 
    yang singkat dan dalam kalimat yang mudah dipahami. Dalam pengembangannya, aplikasi ini memanfaatkan teknologi 
    Large Language Model (LLM) yang mampu memahami konteks bahasa manusia dan menghasilkan teks yang lebih natural.

    Dengan aplikasi ini, kamu bisa dengan mudah membaca thread Twitter tanpa harus membaca satu persatu tweetnya. 
    Untuk menggunakan aplikasi ini, masukkan URL thread Twitter pada kotak input di bawah ini dan klik tombol 
    "Ringkas". Aplikasi akan menampilkan ringkasan dan konten thread asli di bawahnya.
""")

# Set the input box for the Twitter thread URL in Bahasa Indonesia
url = st.text_input(label="Masukkan URL Thread Twitter", placeholder='misalnya https://twitter.com/jack/status/1082159636994121729')

# Add a "Ringkas" button in Bahasa Indonesia with casual tone
if st.button("Ringkas"):
    # Scrape the Twitter thread and summarize the content
    if url:
        scrape_content = scrape_twitter_thread(url)
        summary = get_thread_summary(scrape_content)
    else:
        scrape_content = ""
        summary = ""

    # Display the summary and original thread content
    if summary:
        st.header("Ringkasan Dari Thread Twitter yang Kamu Input:")
        st.write(summary)

    if scrape_content:
        st.header("Konten Dari Thread Asli:")
        st.write(scrape_content)

    if not summary and not scrape_content:
        st.write("Mohon masukkan URL thread Twitter yang valid untuk menampilkan ringkasan.")