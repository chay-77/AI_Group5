Hello All!

This is the Youtube Comment Sentiment Analysis Project!

# YouTube Comment Sentiment Analysis

This project aims to perform sentiment analysis on YouTube comments using natural language processing techniques.

## Overview

YouTube comments are an essential part of user engagement, but they can also contain valuable sentiment information. This project utilizes machine learning models and sentiment analysis techniques to understand the sentiment behind these comments.

## Methodology

The methodology section outlines the systematic approach undertaken in the project, starting with data collection. Pre-labeledcomments from popular YouTube videos, tweets, and blogs form thedataset. Rigorous preprocessing addresses issues like non-alphabeticcharacters and missing values, setting the stage for effective analysis. The application of versatile NLP techniques follows, including tokenization, removal of stop words, lemmatization, and stemming. These processes refine the raw textual data, preparing itfor machine learning analysis. Vectorization, accomplished through both CountVectorizer and TfidfVectorizer, plays a crucial role in transforming textual data into a format suitable for model training. The subsequent model training involves five distinct machine learning models: Naive Bayes, Logistic Regression, SVM, Decision Tree, and Random Forest.

## Features

- **Data Collection:** Utilizes YouTube Data API to fetch comments from specified videos or channels.
- **Preprocessing:** Cleansing and preparing comments for analysis by removing noise, special characters, and stopwords.
- **Sentiment Analysis:** Employs NLP models to classify comments into positive, negative, or neutral sentiments.

## Requirements

- Python 3.x
- Required Python packages (specified in `requirements.txt`)
- YouTube Data API key

## Setup

1. Clone the repository:

2. Install the necessary dependencies:

3. Obtain a YouTube Data API key from the [Google Developers Console](https://console.developers.google.com/) and .

4. Run the analysis script:

## Usage

- Modify `config.py` to specify the YouTube video or channel for comment analysis.
- Adjust preprocessing steps or sentiment analysis models in the `analyze_comments.py` script as needed.
- Run the script to perform analysis and view the results.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up?: sentiment classification using machine learning techniques. Proceedings of the ACL-02 conference on Empirical methods innatural language processing-Volume 10.
- Bird, S., Klein, E., & Loper, E. (2009). Natural language processingwith Python: analyzing text with the natural language toolkit.
- O'Reilly Media, Inc.
- Chen, H., Zhang, J., Xu, B., & Chen, H. (2017). Sentiment analysisusing various machine learning techniques: A review. Computer Science Review, 22, 67-73.

## Contact

For any inquiries or suggestions, please contact [Pallavi Bongu / Sai Chaitanya Kolli / Keerthi Balabhadruni ].

