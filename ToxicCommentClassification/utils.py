import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    pass

class TextPreprocessor:
    def __init__(self, max_words=50000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text):
        """
        Lemmatize text
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words 
                           if word not in self.stop_words and word.isalpha()]
        return ' '.join(lemmatized_words)
    
    def get_sentiment(self, text):
        """
        Get sentiment polarity
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def fit_tokenizer(self, texts):
        """
        Fit tokenizer on texts
        """
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded_sequences
    
    def preprocess_dataset(self, df, text_column='comment_text', label_columns=None):
        """
        Preprocess entire dataset
        """
        if label_columns is None:
            label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Clean text
        print("Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Lemmatize text
        print("Lemmatizing text...")
        df['lemmatized_text'] = df['cleaned_text'].apply(self.lemmatize_text)
        
        # Get sentiment
        print("Calculating sentiment...")
        df['sentiment'] = df['cleaned_text'].apply(self.get_sentiment)
        
        # Fit tokenizer
        print("Fitting tokenizer...")
        self.fit_tokenizer(df['lemmatized_text'].tolist())
        
        # Convert texts to sequences
        print("Converting texts to sequences...")
        X = self.texts_to_sequences(df['lemmatized_text'].tolist())
        
        # Get labels
        y = df[label_columns].values if all(col in df.columns for col in label_columns) else None
        
        return X, y, df
    
    def save_tokenizer(self, filepath):
        """
        Save tokenizer to file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, filepath):
        """
        Load tokenizer from file
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

def load_data(train_path, test_path=None, val_path=None, text_column='comment_text'):
    """
    Load dataset from CSV files
    """
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    test_df = None
    if test_path:
        print(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
    
    val_df = None
    if val_path:
        print(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path)
    elif test_df is not None:
        # Split test data into validation and test
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced dataset
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = []
    n_classes = y.shape[1]
    
    for i in range(n_classes):
        weights = compute_class_weight('balanced', classes=np.unique(y[:, i]), y=y[:, i])
        class_weights.append(weights[1])  # Weight for positive class
    
    return class_weights