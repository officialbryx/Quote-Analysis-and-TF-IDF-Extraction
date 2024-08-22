import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

# Load the CSV file
def load_csv(filename='quotes.csv'):
    return pd.read_csv(filename)

# Clean the text by removing non-alphabetic characters and numbers
def clean_text(text):
    text = text.lower()
    # Remove all non-alphabetic characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Calculate the term frequency (TF) 
def calculate_term_frequency(term, text):
    words = text.split()
    count = words.count(term)
    total_terms = len(words)
    tf = count / total_terms
    return count, total_terms, tf, words

# Calculate the inverse document frequency (IDF) 
def calculate_inverse_document_frequency(term, tfidf_matrix, feature_names):
    term_idx = np.where(feature_names == term)[0]
    if len(term_idx) == 0:
        return 0
    term_idx = term_idx[0]
    num_docs_with_term = np.sum(tfidf_matrix[:, term_idx].toarray() > 0)
    total_docs = tfidf_matrix.shape[0]
    if num_docs_with_term == 0:
        return 0
    return np.log(total_docs / num_docs_with_term)

# Calculate the TF-IDF 
def calculate_tf_idf(term, doc, tfidf_matrix, feature_names):
    count, total_terms, tf, terms_list = calculate_term_frequency(term, doc)
    idf = calculate_inverse_document_frequency(term, tfidf_matrix, feature_names)
    tfidf = tf * idf
    return count, total_terms, tf, idf, tfidf, terms_list

# Extract the most important term for each document
def get_most_important_terms(tfidf_matrix, feature_names):
    important_terms = []
    
    # Iterate over each document
    for i in range(tfidf_matrix.shape[0]):
        # Get the TF-IDF scores for the current document
        scores = tfidf_matrix[i, :].toarray().flatten()
        # Find the index of the highest score
        max_score_idx = scores.argmax()
        # Get the term and score
        term = feature_names[max_score_idx]
        score = scores[max_score_idx]
        important_terms.append((term, score))
    
    return important_terms

# Main Function
def main():
    data = load_csv()
    # Clean the text data
    data['Processed_Text'] = data['Processed_Text'].apply(clean_text)
    documents = data['Processed_Text'].tolist()
    
    # Vectorize the text data with stopwords removal
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    important_terms = get_most_important_terms(tfidf_matrix, feature_names)
    
    # Calculate and print TF, IDF, and TF-IDF for the most important term in each quote
    for i, (term, score) in enumerate(important_terms):
        doc = documents[i]
        count, total_terms, tf, idf, calculated_tfidf, terms_list = calculate_tf_idf(term, doc, tfidf_matrix, feature_names)
        print(f"Quote: {data.iloc[i]['Quote']}")
        print(f"Most Important Term: {term}")
        print(f"Number of '{term}' in quote: {count}")
        print(f"Total number of terms in quote: {total_terms}")
        print(f"Terms in quote: {terms_list}")
        print(f"TF(term, quote) = {tf:.6f}")
        print(f"IDF(term) = {idf:.6f}")
        print(f"TF-IDF = TF * IDF = {calculated_tfidf:.6f}")
        print(f"Stored Term Score: {score:.6f}")
        print("-" * 50)
    
    # Add the important term and its score to the original dataframe
    data['Most_Important_Term'] = [term for term, score in important_terms]
    data['Term_Score'] = [score for term, score in important_terms]
    
    # Print or save the updated dataframe
    print(data)

    # Save the data into a csv file
    data.to_csv('quotes_with_important_terms.csv', index=False)

if __name__ == '__main__':
    main()
