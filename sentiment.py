import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
reviews_df = pd.read_csv('reviews_data.csv')

# Checking for missing values
missing_values = reviews_df.isnull().sum()

# Summary of the distribution of ratings
rating_distribution = reviews_df['Rating'].value_counts(normalize=True) * 100

# Examine the length of reviews
reviews_df['Review_Length'] = reviews_df['Review'].apply(lambda x: len(str(x)
                                                                       .split()))
review_length_stats = reviews_df['Review_Length'].describe()

# Removing rows with missing ratings
reviews_df_cleaned = reviews_df.dropna(subset=['Rating']).copy()

# Checking the shape of the cleaned dataset
new_shape = reviews_df_cleaned.shape

# Transforming ratings into sentiment labels


def rating_to_sentiment(rating):
    if rating in [1.0, 2.0]:
        return 'Negative'
    elif rating == 3.0:
        return 'Neutral'
    else:
        return 'Positive'


reviews_df_cleaned['Sentiment'] = reviews_df_cleaned['Rating'].apply(
    rating_to_sentiment)

# Checking the distribution of the new Sentiment column
sentiment_distribution = reviews_df_cleaned['Sentiment'].value_counts(
    normalize=True) * 100

# Initialize stemmer and set of stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_review(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(
        token) for token in tokens if token.isalpha() and token not in stop_words]

    return ' '.join(tokens)


# Apply preprocessing to the Review column
reviews_df_cleaned['Processed_Review'] = reviews_df_cleaned['Review'].apply(
    preprocess_review)

# Features and target variable
X = reviews_df_cleaned['Processed_Review']
y = reviews_df_cleaned['Sentiment']

# Splitting the data into training and test sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the Logistic Regression classifier
logreg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logreg.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy, classification_rep)
