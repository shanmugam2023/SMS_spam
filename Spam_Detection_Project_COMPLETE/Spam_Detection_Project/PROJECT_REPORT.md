# Project Report: SMS Spam Detection Using Machine Learning

## Aim
To build a machine learning application that classifies SMS messages as spam or not spam.

## Problem statement
Unwanted SMS messages may contain misleading offers, advertisements, or suspicious links. Manually checking every message can be inconvenient.

## Objectives
1. Prepare labeled SMS data.
2. Convert text into numerical features using TF-IDF.
3. Train a Multinomial Naive Bayes classifier.
4. Evaluate it on test data.
5. Build a simple interface to predict new messages.

## Tools
Python, Pandas, Scikit-learn, TF-IDF, Multinomial Naive Bayes, Streamlit.

## Algorithm
1. Start.
2. Load the labeled SMS dataset.
3. Separate messages and labels.
4. Split data into training and testing sets.
5. Convert text into TF-IDF features.
6. Train the Naive Bayes model.
7. Accept a new SMS message.
8. Convert it using the trained vectorizer.
9. Predict spam or ham.
10. Display the prediction.
11. Stop.

## Project management and implementation
The work is divided into planning, dataset preparation, model development, testing, interface implementation, documentation, and GitHub publication.

## Conclusion
This project demonstrates a basic end-to-end machine learning workflow for text classification. The small sample dataset is intended only for learning; predictions are not guaranteed to be correct.
