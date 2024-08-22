import requests
from bs4 import BeautifulSoup
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def scrape_quotes(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    quotes = soup.find_all('div', class_='quote')
    
    data = []
    
    for quote in quotes:
        text = quote.find('span', class_='text').get_text()
        author = quote.find('small', class_='author').get_text()
        tags = [tag.get_text() for tag in quote.find_all('a', class_='tag')]
        
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text)
        filtered_tokens = remove_stopwords(tokens)
        lemmatized_tokens = lemmatize_tokens(filtered_tokens)
        processed_text = ' '.join(lemmatized_tokens)
        
        data.append({
            'Quote': text,
            'Author': author,
            'Tags': ', '.join(tags),
            'Processed_Text': processed_text
        })
    
    return data

# Save to CSV
def save_to_csv(data, filename='quotes.csv'):
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# Main Function
def main():
    url = 'https://quotes.toscrape.com/'
    data = scrape_quotes(url)
    save_to_csv(data)
    
    # Print the first few entries to verify the data
    for entry in data[:5]:  # Print the first 5 quotes for verification
        print(f"Quote: {entry['Quote']}")
        print(f"Author: {entry['Author']}")
        print(f"Tags: {entry['Tags']}")
        print(f"Processed Text: {entry['Processed_Text']}")
        print("-" * 50)

# Run the main function
if __name__ == '__main__':
    main()
