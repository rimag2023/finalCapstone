## Project. Sentiment Analysis.


## What the project does?

 The 'sentiment_analysis.py' script performs sentiment analysis on consumer reviews of Amazon products using the spaCy library. It reads a dataset of product reviews from a CSV file, 
 preprocesses the text data by removing stopwords and performing basic cleaning, and then applies sentiment analysis using spaCy's buid-in capabilities. Finally, it appends the sentiment 
 analysis results to the dataset and prints the first few rows of the dataset with the sentiment analysis results.

## Features

•  Text Processing: 
 1) Tokenization: Splitting text into individual words or tokens.
 2) Stopword Removal: Removing common words that do not add much meaning to the text.
 3) Lemmatization: Converting words to their base or dictionary form.
 4) Text Cleaning: Removing special characters and punctuation.
•  Sentiment Analysis.
 1) Polarity Detection: Positive, negative or neutral sentiment.
 2) Sentiment Intencity: Measuring the strength or intensity of the sentiment expressed in the text.
 3) Subjectivity Analysis: Identifying the degree of subjectivity or opinionatedness in the text.
•  Visualization.
•  Model Evaluation
 1) Accuracy Metrics: Evaluating the performance of the sentiment analysis model using metrics, such as accuracy and precision.
• Customization Options:
 1) Custom Stopword Lists: Allowing users to customize the list of stopwords based on their specific domain or requirements.
 2) Fine-Tuning Models: Providing options to fine-tune the pre-trained spaCy model for better performance on specific datasets or tasks.
• Integration.
• Scalability.
• Documentation.

## Prerequisites

• Python 3.12.2. You can download and install Python from the official website: https://www.python.org/downloads/
• pip: Ensure that you have pip, the Python package manager, installed. It usually comes bundled with Python installations. You can check, if you have got pip installed on your machine 
by running 'pip --version' in your command line or terminal.
• spaCy:Install the spaCy library, which provides NLP functionalities including tokenization, part-of-speech tagging, and named entity recognition. You can install spaCy using pip:
pip install spaCy
• spaCy Model: Download and install a spaCy model, such as 'en_core_web_sm', which is a small English model that includes vocabulary, syntax, and entities. You can download it using 
the following command:
python -m spacy download en_core_web_sm
• 1) TextBlob. Install using pip:
     pip install textblob
  2) Additionally, you may need to download the necessary NLTK corpora by running the following Python code:
     import nltk
     nltk.download('punk')
     nltk.download('averaged_perceptron_tagger')
     nltk.download('wordnet')
• Dataset: Prepare a dataset of product reviews in CSV format. You can collect this data from various sources such as Kaggle or directly from Amazon's datasets if available.
     
  

## Installation

Clone the repository: bash git clone https://github.com/rimag2023/finalCapstone.git

Navigate to the project directory: bash cd finalCapstone

Install dependencies: pip install -r requirements.txt (optional)

Run the application: 'sentiment_analysis_3.py'

## Database

Python 3.12.2


## Who maintains the project?

The author.

## Contributing.

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## Acknowledgement

Inspired by sentiment_analysis by doing research online and working on my own own project. https://github.com/rimag2023/finalCapstone/edit/main/README.md
