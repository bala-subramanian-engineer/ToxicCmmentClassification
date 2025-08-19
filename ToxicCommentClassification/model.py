import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_toxic_comment_model(vocab_size, embedding_dim=100, max_len=200, num_classes=6):
    """
    Create Bi-LSTM model for toxic comment classification
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of embedding layer
        max_len (int): Maximum sequence length
        num_classes (int): Number of output classes
    
    Returns:
        model: Compiled Bi-LSTM model
    """
    # Input layer
    input_layer = layers.Input(shape=(max_len,))
    
    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True
    )(input_layer)
    
    # Spatial Dropout to prevent overfitting
    dropout_embed = layers.SpatialDropout1D(0.2)(embedding)
    
    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(layers.LSTM(
        64, 
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ))(dropout_embed)
    
    lstm2 = layers.Bidirectional(layers.LSTM(
        32, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ))(lstm1)
    
    # Dense layers
    dense1 = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm2)
    dropout1 = layers.Dropout(0.5)(dense1)
    
    dense2 = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)
    
    # Output layer (multi-label classification)
    output_layer = layers.Dense(num_classes, activation='sigmoid')(dropout2)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer and metrics
    
    Args:
        model: Keras model to compile
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        model: Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def create_attention_model(vocab_size, embedding_dim=100, max_len=200, num_classes=6):
    """
    Create Bi-LSTM model with attention mechanism
    """
    # Input layer
    input_layer = layers.Input(shape=(max_len,))
    
    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True
    )(input_layer)
    
    # Spatial Dropout
    dropout_embed = layers.SpatialDropout1D(0.2)(embedding)
    
    # Bidirectional LSTM
    bilstm = layers.Bidirectional(layers.LSTM(
        64, 
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ))(dropout_embed)
    
    # Attention mechanism
    attention = layers.Attention()([bilstm, bilstm])
    attention = layers.GlobalAveragePooling1D()(attention)
    
    # Dense layers
    dense1 = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(attention)
    dropout1 = layers.Dropout(0.5)(dense1)
    
    dense2 = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)
    
    # Output layer
    output_layer = layers.Dense(num_classes, activation='sigmoid')(dropout2)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model