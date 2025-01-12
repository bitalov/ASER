
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on
        # position and dimension
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:, :x.size(1)]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        attn_output = torch.sum(x * attn_weights, dim=1)     # (batch_size, input_dim)
        return attn_output

class EmotionRecognitionModelTransfomer(nn.Module):
    def __init__(self, num_emotions, input_height):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv_block = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),
        )

        # Calculate the output height after convolution and pooling layers
        conv_output_height = input_height // (2 ** 3)  # Divided by 8 due to three MaxPool layers

        # Fully connected layer to reduce dimensionality before Transformer
        self.fc1 = nn.Linear(64 * conv_output_height, 128)
        self.dropout_fc1 = nn.Dropout(p=0.3)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=128, max_len=5000)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention Pooling Layer
        self.attention_pooling = AttentionPooling(input_dim=128)

        # Output layer
        self.fc2 = nn.Linear(128, num_emotions)

    def forward(self, x):
        # x shape: (batch_size, 1, freq_bins, time_steps)

        batch_size = x.size(0)

        # CNN feature extraction
        x = self.conv_block(x)
        # x shape: (batch_size, channels, freq_bins', time_steps')

        # Prepare for Transformer
        x = x.permute(0, 3, 1, 2)  # Shape: (batch_size, time_steps', channels, freq_bins')
        x = x.contiguous().view(batch_size, x.size(1), -1)  # Flatten channels and freq_bins'

        # Fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)  # Shape: (batch_size, time_steps', 128)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer expects input of shape (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # Shape: (time_steps', batch_size, 128)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # Output shape: (time_steps', batch_size, 128)

        # Transpose back to (batch_size, time_steps', embedding_dim)
        x = x.permute(1, 0, 2)

        # Attention pooling
        x = self.attention_pooling(x)  # Shape: (batch_size, 128)

        # Output layer
        logits = self.fc2(x)

        return logits  # Return logits directly

