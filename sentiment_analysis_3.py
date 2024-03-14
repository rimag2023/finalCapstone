# Import necessary libraries
import pandas as pd
import spacy
from textblob import TextBlob
import re


# Load the spaCy English model
# 'en_core_web_sm' assigns context-specific token vectors, POS tags, dependency parse, and named entities.
nlp = spacy.load('en_core_web_sm')


# Load the dataset into a pandas DataFrame
dataframe = pd.read_csv('amazon_product_reviews.csv', low_memory=False)


# Display the first few reviews and the total number of reviews
print(dataframe['reviews.text'].head())
print(dataframe['reviews.text'].shape)


# Drop rows where reviews are missing and print the new total
clean_data = dataframe.dropna(subset=['reviews.text'])
reviews_data = clean_data['reviews.text']
print(reviews_data.shape)


# Define a function to preprocess review text
def preprocess_text(text):
   """
   Preprocesses the input text by cleaning and lemmatizing it.
   This function performs several preprocessing steps on the input text:
   1. Removes special characters using a regular expression, keeping only words and spaces.
   2. Uses the spaCy NLP library to process the cleaned text, performing tokenization.
   3. Generates a list of lemmatized tokens from the processed text, excluding stopwords and punctuation.
   4. Concatenates the lemmatized tokens back into a single string.
   Parameters:
   - text (str): The input text string to be preprocessed.
   Returns:
   - str: The preprocessed and cleaned text, with lemmatized tokens joined into a single string.
   """
   # Remove special characters
   pattern = r'[^\w\s]'
   cleaned_text = re.sub(pattern, '', text)


   # Process the text with spaCy
   doc = nlp(cleaned_text)


   # Initialise an empty list to store clean tokens
   clean_tokens = []


   # Loop through each token in the document
   for token in doc:
       # Check if the token is not a stop word, not punctuation, and not whitespace
       if not token.is_stop and not token.is_punct and token.text.strip():
           # Lemmatize the token, convert it to lowercase, and add it to the list
           clean_token = token.lemma_.lower()
           clean_tokens.append(clean_token)


   # Join the clean tokens back into a single string
   clean_text = ' '.join(clean_tokens)


   return clean_text


# Apply preprocessing to clean the reviews
clean_reviews = reviews_data.apply(preprocess_text)


# Display the first few preprocessed reviews
print(clean_reviews.head())


# Define a function for sentiment analysis using TextBlob
def sentiment_analysis(reviews):
   """
   Analyses the sentiment polarity of a given text review.
   This function utilises TextBlob to compute the sentiment polarity of the input review text.
   The polarity score is a float within the range [-1.0, 1.0] where -1.0 signifies a negative
   sentiment, 1.0 signifies a positive sentiment, and values around 0 represent neutral sentiments.
   Parameters:
   - review (str): The review text whose sentiment polarity is to be analysed.
   Returns:
   - float: The polarity score of the review, indicating the sentiment of the text.
   """
   # Analyse sentiment polarity with TextBlob
   textblob_review = TextBlob(reviews)
   polarity = textblob_review.polarity
   return polarity


# Apply sentiment analysis to preprocessed reviews and store results in a new 'polarity' column
clean_data['polarity'] = clean_reviews.apply(sentiment_analysis)


# Display the original reviews with their corresponding polarity scores
print(clean_data[['reviews.text', 'polarity']].head())


# Load a larger spaCy model for more detailed processing and similarity comparison
nlp = spacy.load('en_core_web_md')


# Prompt user for indices of reviews to compare for similarity
index_a = int(input('Please enter the index of the first review you want to compare: '))
index_b = int(input('Please enter the index of the second review you want to compare: '))


# Ensure user-selected indices are valid
num_reviews = len(reviews_data)
if index_a < 0 or index_a >= num_reviews or index_b < 0 or index_b >= num_reviews:
   print('\n',"Error: Invalid index. Please enter indices within the range 0 to", num_reviews - 1)
else:
   # Retrieve and display the selected reviews
   review_a = reviews_data.iloc[index_a]
   review_b = reviews_data.iloc[index_b]


   print("Review A:", review_a, '\n')
   print("Review B:", review_b, '\n')


   # Calculate and display the similarity score between the two reviews
   doc_a = nlp(review_a)
   doc_b = nlp(review_b)   
   similarity_score = doc_a.similarity(doc_b)
   print(f'Similarity score of the two reviews: {round(similarity_score,3)}')


# Additional interactive component for testing sentiment analysis on a set number of reviews
selection_num = int(input("Enter the number of reviews you want to test"))
# Select and display sentiment scores for the first 'n' reviews specified by the user
sample_reviews = clean_reviews.head(selection_num) 
for review in sample_reviews:
   sentiment_score = sentiment_analysis(review)
   print("Review:", review)
   print("Sentiment Score:", sentiment_score)
   print()

