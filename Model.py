import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Path to the directory containing your text files
directory_path = directory_path = '/Users/nabeel/Documents/D-alp/Training Dataset'


# Categories you're interested in
categories = ['Legal', 'Medical', 'Financial', 'Educational', 'Business', 'News', 'Technical', 'Creative', 'Scientific', 'Government']

# Initialize lists to store data
texts = []
labels = []

# Read data from files
for category in categories:
    file_path = os.path.join(directory_path, f'{category.lower()}.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
        # Assuming each document is separated by a newline
        documents = text_data.split('\n')
        for doc in documents:
            if doc.strip():  # Ensure document is not empty
                texts.append(doc)
                labels.append(category)

# Create a DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Define a machine learning pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
