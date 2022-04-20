"""
File: models.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    - This file contains the class definitions for the different models
    - For a triplet network, the model is defined as:
        - Input: (img, positive_text, negative_text)
"""
import torch
import torch.nn as nn


# Network definition for the textual aggregation
class EmbeddingTextNet(nn.Module):
    def __init__(self, embedding_size, output_size, late_fusion=None):
        super(EmbeddingTextNet, self).__init__()
        self.late_fusion = late_fusion
        # Define a fully connected layer with input of n_input and output n_output neurons
        self.fc1 = nn.Sequential(nn.Linear(embedding_size, 1024),
                                 nn.PReLU(),
                                 nn.Linear(1024, 2048),
                                 nn.PReLU(),
                                 nn.Linear(2048, output_size)
                                 )  # 256 is the size of the final text embedding

        # Define a dropout layer with probability p
        self.dropout = nn.Dropout(p=0.1)

        # Define a transformers to aggregate the text embeddings

        # Define a LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, num_layers=1, batch_first=True)

    # forward method
    def forward(self, x):
        # Apply late fusion aggregation
        if self.late_fusion is not None:
            # Apply late fusion aggregation
            x = self.late_fusion(self.late_fusion, x)

        # Project to common latent space for the image and text
        out = self.fc1(x)

        # Dropout bc why not
        out = self.dropout(out)
        return out

    def text_aggregation(self, text_aggregation_type, x):
        # Aggregate the text embeddings

        if text_aggregation_type == 'lstm':
            # Apply LSTM
            _, (h_n, c_n) = self.lstm(x)
            aggregated = h_n[-1]
        elif text_aggregation_type == 'attention':
            # Apply attention
            pass
        elif text_aggregation_type == 'self_attention':
            aggregated = self.self_attention(x)

        return aggregated


# Network definition for the image embedding
class EmbeddingImageNet(nn.Module):
    def __init__(self, output_size):
        super(EmbeddingImageNet, self).__init__()

        # Define a fully connected layer with input of n_input and output n_output neurons
        self.fc1 = nn.Linear(4096, output_size)  # 512 is the size of the final image embedding

        # Define a dropout layer with probability p
        self.dropout = nn.Dropout(p=0.5)

    # forward method
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        return out


# Network definition for the triplet network in the image to text case
class TripletImageText(nn.Module):
    def __init__(self, embedding_text_net, embedding_image_net, margin=1.0):
        super(TripletImageText, self).__init__()
        self.embedding_text_net = embedding_text_net
        self.embedding_image_net = embedding_image_net
        self.margin = margin

    def forward(self, img, positive_text, negative_text):
        # Get the embeddings for the image and the text
        img_embedding = self.embedding_image_net(img)
        text_embedding = self.embedding_text_net(positive_text)
        negative_text_embedding = self.embedding_text_net(negative_text)

        return img_embedding, text_embedding, negative_text_embedding


# Network definition for the triplet network in the text to image case
class TripletTextImage(nn.Module):
    def __init__(self, embedding_text_net, embedding_image_net, margin=1.0):
        super(TripletTextImage, self).__init__()
        self.embedding_text_net = embedding_text_net
        self.embedding_image_net = embedding_image_net
        self.margin = margin

    def forward(self, text, img1, img2):
        # Get the embeddings for the image and the text

        text_embedding = self.embedding_text_net(text)
        img1_embedding = self.embedding_image_net(img1)
        img2_embedding = self.embedding_image_net(img2)

        return text_embedding, img1_embedding, img2_embedding
