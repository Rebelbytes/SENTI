import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset

app = Flask(__name__)

# ---------------- Dataset and Model ----------------
def load_and_prepare_dataset():
    dataset = load_dataset("prasadsawant7/sentiment_analysis_preprocessed_dataset", split='train')
    df = pd.DataFrame(dataset)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

df = load_and_prepare_dataset()
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- YouTube Comments ----------------
def video_comments(video_id):
    youtube = build('youtube', 'v3', developerKey='AIzaSyCcBrt-UsC9soIxO-y5wu3z8xGzyuu2rIE')
    comments = []
    next_page_token = None
    while True:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except HttpError as e:
            print(f"HTTP Error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    return comments

def get_video_title(video_id):
    youtube = build('youtube', 'v3', developerKey='AIzaSyCcBrt-UsC9soIxO-y5wu3z8xGzyuu2rIE')
    try:
        response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        if response['items']:
            return response['items'][0]['snippet']['title']
        else:
            return "Video Title Not Found"
    except HttpError as e:
        print(f"HTTP Error: {e}")
        return "Video Title Not Found"
    except Exception as e:
        print(f"Error: {e}")
        return "Video Title Not Found"

# ---------------- Sentiment Analysis ----------------
def analyze_youtube_video_sentiment(video_id):
    comments = video_comments(video_id)
    if not comments:
        return None
    cleaned_comments = [clean_text(c) for c in comments]
    X_comments = tfidf.transform(cleaned_comments).toarray()
    predicted_sentiments = model.predict(X_comments)
    
    # Sentiment percentages
    positive, negative, neutral = np.sum(predicted_sentiments==2), np.sum(predicted_sentiments==0), np.sum(predicted_sentiments==1)
    total = len(predicted_sentiments)
    positive_pct = (positive/total)*100
    negative_pct = (negative/total)*100
    neutral_pct = (neutral/total)*100

    results = {
        'positive_percentage': round(positive_pct, 2),
        'negative_percentage': round(negative_pct, 2),
        'neutral_percentage': round(neutral_pct, 2),
        'positive_comments': [comments[i] for i,s in enumerate(predicted_sentiments) if s==2][:10],
        'negative_comments': [comments[i] for i,s in enumerate(predicted_sentiments) if s==0][:10],
        'neutral_comments': [comments[i] for i,s in enumerate(predicted_sentiments) if s==1][:10]
    }
    return results

# ---------------- Pie Chart ----------------
def plot_sentiment_pie_chart(results):
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [results['positive_percentage'], results['negative_percentage'], results['neutral_percentage']]
    colors = ['#99ff99', '#ff6666', '#66b3ff']
    explode = (0.05, 0.05, 0.05)
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode, shadow=True)
    plt.title('Sentiment Distribution')
    path = os.path.join('static', 'sentiment_pie_chart.png')
    plt.savefig(path)
    plt.close()
    return path

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form['video_id'].split('v=')[-1]
    results = analyze_youtube_video_sentiment(video_id)
    if not results:
        return "No comments found or invalid video ID"
    video_title = get_video_title(video_id)
    pie_path = plot_sentiment_pie_chart(results)
    return render_template('senti_result.html', video_id=video_id, video_title=video_title, results=results, pie_chart_url=pie_path)

if __name__ == "__main__":
    app.run(debug=True)
