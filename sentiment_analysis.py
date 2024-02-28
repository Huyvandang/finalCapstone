# Develop a script for sentiment analysis.
# Implement a sentiment analysis model using spaCy.
# Load en_core_web_sm model to enable NLP task. 

# Preprocess the text data:
# Remove stopwords and perform necessary text cleaning.
# Use 'reviews_data = dataframe['review.text']' to select the 'review.text' column.
# Remove all missing values from the column using 'clean_data = dataframe.dropna(subset=['reviews.text'])' function from Pandas. 

# Create a function for sentiment analysis:
# Define a function that takes a product review as input and predicts its sentiment.

# Test the model on sample product reviews:
# Test the sentiment analysis function on a few sample product reviews to verify its accuracy in predicting sentiment.

# Choose two product reviews from 'review.text' column and compare their similarity. 
# To select a specific review, use indexing with the code 'my_review_of_choice = data['reviews.text'][0]' 

# Import necessary libraries
import pandas as pd
from os import chdir, path
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the English model and add SpacyTextBlob extension for sentiment analysis
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Change the current working directory to the directory of the script
chdir(path.dirname(__file__))

# Load the dataset into a DataFrame
file_path = 'amazon_product_reviews.csv'
dataframe = pd.read_csv(file_path, dtype={
    # Specify the data types for each column to avoid any inference issues
    'id': str, 'name': str, 'asins': str, 'brand': str, 'categories': str, 'keys': str, 'manufacturer': str, 
    'reviews.date': str, 'reviews.dateAdded': str, 'reviews.dateSeen': str, 'reviews.didPurchase': str, 
    'reviews.doRecommend': str, 'reviews.id': str, 'reviews.numHelpful': str, 'reviews.rating': str, 
    'reviews.sourceURLs': str, 'reviews.text': str, 'reviews.title': str, 'reviews.userCity': str, 
    'reviews.userProvince': str, 'reviews.username': str, })

# Select the 'reviews.text' column
reviews_data = dataframe['reviews.text']

# Remove missing values from the 'reviews.text' column
clean_data = dataframe.dropna(subset=['reviews.text'])

# Define a function for text preprocessing - removing stopwords and cleaning
def preprocess_text(text):
    doc = nlp(text)
    
    # Remove stopwords and non-alphabetic tokens
    cleaned_tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    # Join the cleaned tokens back into a string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

# Apply text preprocessing to the 'reviews.text' column and store the result in a new column 'processed_reviews'
clean_data = clean_data.copy()
clean_data['processed_reviews'] = clean_data['reviews.text'].apply(preprocess_text)

# Display the cleaned data
print("Cleaned Data:")
print(clean_data[['reviews.text', 'processed_reviews']].head())

# Print a separator line for better visibility
print("\n" + "=" * 80 + "\n")

# Define a function for sentiment analysis
def predict_sentiment(review):
    # Apply SpaCy's sentiment analysis
    doc = nlp(review)
    
    # Access the sentiment polarity from the SpaCyTextBlob extension
    sentiment_polarity = doc._.polarity

    # Classify the sentiment based on polarity
    if sentiment_polarity > 0:
        sentiment = 'positive'
    elif sentiment_polarity == 0:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'

    return sentiment
    
# Loop through rows 0 to 98 and test the sentiment analysis function.
for i in range(99):
    text_review = dataframe.at[i, 'reviews.text']
    result = predict_sentiment(text_review)
    print(f"Sentiment score for row {i}: {result}")