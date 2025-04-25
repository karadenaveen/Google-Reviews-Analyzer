import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Ensure NLTK is ready
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv(r"C:\Users\NAVEENKUMAR\Documents\Sentimental_Analysis\Reviews.csv")
df['date'] = pd.to_datetime(df['Time'], unit='s')  # Convert UNIX time to datetime
df['review'] = df['Text']  # Rename for consistency

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = nltk.word_tokenize(str(text).lower())
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

df['clean_review'] = df['review'].apply(clean_text)

# Sentiment using TextBlob
df['polarity'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Time trends by week
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
sentiment_trend = df.groupby(['week', 'sentiment']).size().unstack().fillna(0)

# Plot sentiment trend
plt.figure(figsize=(12,6))
sentiment_trend.plot(marker='o')
plt.title("Sentiment Trend Over Time")
plt.xlabel("Week")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Topic Modeling (LDA)
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['clean_review'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

def print_topics(model, vectorizer, top_n=10):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx + 1}: ", [words[i] for i in topic.argsort()[-top_n:]])

print("\n🔍 Top Themes from Reviews:")
print_topics(lda, vectorizer)

# Weekly summaries
def generate_summary(df):
    summary = {}
    for week, group in df.groupby('week'):
        top_phrases = group['clean_review'].str.split().explode().value_counts().head(5).index.tolist()
        summary[str(week)] = {
            'total_reviews': len(group),
            'positive': (group['sentiment'] == 'positive').sum(),
            'negative': (group['sentiment'] == 'negative').sum(),
            'neutral': (group['sentiment'] == 'neutral').sum(),
            'top_phrases': top_phrases
        }
    return summary

weekly_report = generate_summary(df)
for week, data in weekly_report.items():
    print(f"\n {week} Summary:")
    print(f"Total: {data['total_reviews']},  {data['positive']},  {data['negative']},  {data['neutral']}")
    print(f"Top phrases: {', '.join(data['top_phrases'])}")

# Flag complaints
complaint_keywords = ['bad', 'worst', 'terrible', 'disappointed', 'rude', 'dirty']
df['is_complaint'] = df['clean_review'].apply(lambda text: any(kw in text for kw in complaint_keywords))

# Basic category classification
def classify_review(text):
    if 'service' in text:
        return 'service'
    elif 'location' in text or 'place' in text:
        return 'location'
    elif 'product' in text or 'item' in text:
        return 'product'
    else:
        return 'other'

df['category'] = df['clean_review'].apply(classify_review)

# # Save final output
# df.to_csv("processed_reviews.csv", index=False)
