from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils import TextPreprocessor

app = Flask(__name__)

# Configuration
MODEL_PATH = 'toxic_comment_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_LEN = 200
THRESHOLD = 0.5

# Load model and tokenizer
def load_toxic_model():
    global model, preprocessor
    try:
        model = load_model(MODEL_PATH)
        preprocessor = TextPreprocessor(max_len=MAX_LEN)
        preprocessor.load_tokenizer(TOKENIZER_PATH)
        print("Toxic comment model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        preprocessor = None

load_toxic_model()

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def preprocess_text(text):
    cleaned_text = preprocessor.clean_text(text)
    lemmatized_text = preprocessor.lemmatize_text(cleaned_text)
    return lemmatized_text

@app.route('/classify', methods=['POST'])
def classify_toxic():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    threshold = data.get('threshold', THRESHOLD)
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Convert to sequence
        sequence = preprocessor.texts_to_sequences([processed_text])
        
        # Make prediction
        predictions = model.predict(sequence)
        
        # Format results
        results = {}
        for i, class_name in enumerate(class_names):
            results[class_name] = {
                'probability': float(predictions[0][i]),
                'is_toxic': bool(predictions[0][i] > threshold)
            }
        
        # Overall toxicity
        max_prob = max([results[cls]['probability'] for cls in class_names])
        results['overall'] = {
            'is_toxic': max_prob > threshold,
            'max_probability': max_prob
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400
    
    texts = data['texts']
    threshold = data.get('threshold', THRESHOLD)
    
    results = []
    for text in texts:
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Convert to sequence
            sequence = preprocessor.texts_to_sequences([processed_text])
            
            # Make prediction
            predictions = model.predict(sequence)
            
            # Format results
            result = {'text': text}
            for i, class_name in enumerate(class_names):
                result[class_name] = {
                    'probability': float(predictions[0][i]),
                    'is_toxic': bool(predictions[0][i] > threshold)
                }
            
            # Overall toxicity
            max_prob = max([result[cls]['probability'] for cls in class_names])
            result['overall'] = {
                'is_toxic': max_prob > threshold,
                'max_probability': max_prob
            }
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'text': text,
                'error': str(e)
            })
    
    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None and preprocessor is not None:
        return jsonify({'status': 'healthy'})
    else:
        return jsonify({'status': 'unhealthy'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)