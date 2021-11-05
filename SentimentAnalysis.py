from tweepy import OAuthHandler
import tweepy
from textblob import TextBlob
from matplotlib import pyplot as plt
import re
import csv


# Twitter credentials for the app

consumer_key = "brWuwU0cLgrMVse0SiqftiNWa"
consumer_secret = "HjLUxe7OUNqTkZfwBkb2IYVjvoBv9EhqOhu36Fdf8BjX3yBwf0"
access_key = "2347130604-6PoRuSsjplj4gsFXk8tmd8wPyuuIwbQGpDUq6fH"
access_secret = "VRq57J0EJ27bIswND5y32o5yL2343T916XZITsVOEmbBQ"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)

search_item = input("Enter the topic of tweets: ")
no_of_tweets = int(input("Enter the number of tweets to search for: "))
tweets = tweepy.Cursor(api.search, q=search_item, lang="en").items(no_of_tweets)

# Defining vairables to compute sentiment
sad = 0
happy = 0
unhappy = 0
neutral = 0



sentiments = [sad, happy, unhappy, neutral]

tweetText = []

# Open/create a file to append data to
csvFile = open('TweetExtracted.csv', 'a')

# Use csv writer
csvWriter = csv.writer(csvFile)
def cleanTweet(tweet):
    # Remove Links, Special Characters etc from tweet
    return " ".join(
        re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet
        ).split()
    )
    


for tweet in tweets:
    # Append to temp so that we can store in csv later. I use encode UTF-8
    tweetText.append(cleanTweet(tweet.text).encode('utf-8'))
    analysis = TextBlob(tweet.text)
    if analysis.sentiment.polarity == 0:
        neutral += 1
    elif -0.5 < analysis.sentiment.polarity < 0:
        sad += 1
    elif analysis.sentiment.polarity <= -0.5:
        unhappy += 1
    else:
        happy += 1






csvWriter.writerow(tweetText)
csvFile.close()


stringofEmotions = ["sad", "happy", "unhappy", "neutral"]
Emotions = [sad, happy, unhappy, neutral]
for x, y in zip(stringofEmotions, Emotions):
    print(x, y)
sentiments2 = []
colors = ["lightgreen", "darkgreen", "lightgreen", "lightsalmon"]
sentiments = [sad, happy, unhappy, neutral]

def percentage(part, whole):

    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

happy = percentage(happy,no_of_tweets)
sad = percentage(sad,no_of_tweets)
unhappy = percentage(unhappy,no_of_tweets)
neutral = percentage(neutral,no_of_tweets)

def pieplot(x):
    labels = [
        "sad [" + str(sad) + "%]",
        "neutral [" + str(neutral) + "%]",
        "unhappy[" + str(unhappy) + "%]",
        "happy [" + str(happy) + "%]",
    ]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    plt.pie(x, explode=(0, 0.05, 0.05, 0))
    plt.title("Pie Chart representing Sentiments")
    plt.legend(labels=labels)
    plt.tight_layout()
    plt.show()
pieplot(x = sentiments)

