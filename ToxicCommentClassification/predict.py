import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import TextPreprocessor

class ToxicCommentClassifier:
    def __init__(self, model_path, tokenizer_path, max_len=200, threshold=0.5):
        """
        Initialize toxic comment classifier
        
        Args:
            model_path (str): Path to trained model
            tokenizer_path (str): Path to tokenizer
            max_len (int): Maximum sequence length
            threshold (float): Confidence threshold for classification
        """
        self.model = load_model(model_path)
        self.preprocessor = TextPreprocessor(max_len=max_len)
        self.preprocessor.load_tokenizer(tokenizer_path)
        self.threshold = threshold
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def preprocess_text(self, text):
        """
        Preprocess a single text
        """
        cleaned_text = self.preprocessor.clean_text(text)
        lemmatized_text = self.preprocessor.lemmatize_text(cleaned_text)
        return lemmatized_text
    
    def predict(self, text):
        """
        Predict toxicity of a comment
        
        Args:
            text (str): Comment text
        
        Returns:
            dict: Prediction results with probabilities for each class
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.preprocessor.texts_to_sequences([processed_text])
        
        # Make prediction
        predictions = self.model.predict(sequence)
        
        # Format results
        results = {}
        for i, class_name in enumerate(self.class_names):
            results[class_name] = {
                'probability': float(predictions[0][i]),
                'is_toxic': bool(predictions[0][i] > self.threshold)
            }
        
        # Overall toxicity
        max_prob = max([results[cls]['probability'] for cls in self.class_names)
        results['overall'] = {
            'is_toxic': max_prob > self.threshold,
            'max_probability': max_prob
        }
        
        return results
    
    def predict_batch(self, texts):
        """
        Predict toxicity for multiple comments
        
        Args:
            texts (list): List of comment texts
        
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                result['text'] = text
                results.append(result)
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Classify toxic comments')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--input_file', type=str, help='File with texts to classify (one per line)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ToxicCommentClassifier(
        args.model_path, 
        args.tokenizer_path, 
        args.max_len, 
        args.threshold
    )
    
    # Get texts to classify
    texts = []
    if args.text:
        texts = [args.text]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Please provide either --text or --input_file argument")
        return
    
    # Make predictions
    if len(texts) == 1:
        results = classifier.predict(texts[0])
        results['text'] = texts[0]
    else:
        results = classifier.predict_batch(texts)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print results
    if len(texts) == 1:
        print(f"Text: {texts[0]}")
        print(f"Overall toxicity: {results['overall']['is_toxic']} (max probability: {results['overall']['max_probability']:.4f})")
        print("\nDetailed predictions:")
        for class_name in classifier.class_names:
            prob = results[class_name]['probability']
            is_toxic = results[class_name]['is_toxic']
            print(f"  {class_name}: {prob:.4f} {'(TOXIC)' if is_toxic else ''}")
    else:
        for result in results:
            if 'error' in result:
                print(f"Error processing text: {result['text']} - {result['error']}")
            else:
                overall = result['overall']
                print(f"Text: {result['text'][:50]}... - Toxic: {overall['is_toxic']} (max prob: {overall['max_probability']:.4f})")

if __name__ == "__main__":
    main()