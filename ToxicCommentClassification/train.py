import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from model import create_toxic_comment_model, compile_model, create_attention_model
from utils import TextPreprocessor, load_data, calculate_class_weights

def train_toxic_model(train_path, model_save_path, test_path=None, val_path=None, 
                     max_words=50000, max_len=200, embedding_dim=100, epochs=20, 
                     batch_size=64, use_attention=False):
    """
    Train the toxic comment classification model
    
    Args:
        train_path (str): Path to training data CSV
        model_save_path (str): Path to save the trained model
        test_path (str): Path to test data CSV
        val_path (str): Path to validation data CSV
        max_words (int): Maximum vocabulary size
        max_len (int): Maximum sequence length
        embedding_dim (int): Embedding dimension
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        use_attention (bool): Whether to use attention mechanism
    """
    # Load data
    train_df, val_df, test_df = load_data(train_path, test_path, val_path)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_words=max_words, max_len=max_len)
    
    # Preprocess training data
    print("Preprocessing training data...")
    X_train, y_train, train_df_processed = preprocessor.preprocess_dataset(
        train_df, 
        label_columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    )
    
    # Preprocess validation data if available
    if val_df is not None:
        print("Preprocessing validation data...")
        X_val, y_val, _ = preprocessor.preprocess_dataset(
            val_df, 
            label_columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        )
    else:
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Preprocess test data if available
    if test_df is not None:
        print("Preprocessing test data...")
        X_test, y_test, _ = preprocessor.preprocess_dataset(
            test_df, 
            label_columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    if test_df is not None:
        print(f"Test data shape: {X_test.shape}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Create model
    vocab_size = min(max_words, len(preprocessor.tokenizer.word_index)) + 1
    num_classes = y_train.shape[1]
    
    if use_attention:
        model = create_attention_model(vocab_size, embedding_dim, max_len, num_classes)
        print("Using attention model")
    else:
        model = create_toxic_comment_model(vocab_size, embedding_dim, max_len, num_classes)
        print("Using Bi-LSTM model")
    
    model = compile_model(model)
    
    # Display model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('training_log.csv')
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate on test set if available
    if test_df is not None:
        print("Evaluating on test set...")
        test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
            X_test, y_test, verbose=0
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
    
    # Save tokenizer
    preprocessor.save_tokenizer('tokenizer.pkl')
    print("Tokenizer saved to tokenizer.pkl")
    
    return model, history, preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train toxic comment classification model')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--model_path', type=str, default='toxic_comment_model.h5', 
                        help='Path to save trained model')
    parser.add_argument('--test_path', type=str, help='Path to test data CSV')
    parser.add_argument('--val_path', type=str, help='Path to validation data CSV')
    parser.add_argument('--max_words', type=int, default=50000, help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
    
    args = parser.parse_args()
    
    # Train model
    train_toxic_model(
        train_path=args.train_path,
        model_save_path=args.model_path,
        test_path=args.test_path,
        val_path=args.val_path,
        max_words=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_attention=args.attention
    )