import re
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from tqdm import tqdm

class word2vecSG(nn.Module):
    """Class to build SWord2Vec from scratch"""

    def __init__(self, text_corpus, window=3, embedding_dim = 10, learning_rate=0.1):
        super(word2vecSG, self).__init__()
        self.window = window
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.text_corpus = text_corpus

        self.vocab, self.word_to_idx, self.idx_to_word = self.build_vocab()
        self.embeddings = nn.Embedding(len(self.vocab), self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, len(self.vocab))
        self.optimizer = SGD(self.parameters(), lr=self.learning_rate)


    def tokenize(self):
        """Tokenizes sentences"""
        tokens = []
        for sentence in tqdm(self.text_corpus, total=len(self.text_corpus), \
                             desc="Tokenizing sentence"):
            words = re.findall(r'\b\w+\b', sentence.lower())
            tokens.extend(words)
        return tokens

    def build_vocab(self):
        "Build a word to index and index to word map"
        self.tokens = self.tokenize()
        #Preserves the prder for the context words  - dict.fromkeys is removing duplicates
        vocab = list(dict.fromkeys(self.tokens)) 
        word_to_idx = {word:i for i,word in enumerate(vocab)}
        idx_to_word = {i:word for i,word in enumerate(vocab)}
        return vocab, word_to_idx, idx_to_word
    
    def generate_training_data(self):
        """Generate target-context pairs based on window size"""
        training_data = []
        for i, target_token in tqdm(enumerate(self.tokens), total = len(self.tokens), desc="Generating training data"):
            target_idx = self.word_to_idx[target_token]
            context_start = max(0,i - self.window)
            context_end = min(len(self.tokens), i + self.window + 1)
            for j in range(context_start, context_end):
                if i!= j:
                    training_data.append( ( target_idx, self.word_to_idx[self.tokens[j]] ) )
        return training_data
    
    def forward(self, target):
        """
        Forward pass: given a target word, get its embedding, apply the linear transformation
        """
        target_embed = self.embeddings(target)  # Get word vector
        output = self.output_layer(target_embed)  # Linear transformation
        return output
    
    def fit(self, epochs = 10):
        """Train the model using SGD"""

        training_data = self.generate_training_data()
        crit = nn.CrossEntropyLoss()
        loss_values = []
        for epoch in range(epochs):
            total_loss = 0
            for target_idx, context_idx in training_data:
                #Convert to tensor
                target_idx = torch.tensor([target_idx])  # Convert to tensor
                context_idx = torch.tensor([context_idx])

                #Forward pass
                self.optimizer.zero_grad()
                output = self.forward(target_idx)
                loss = crit(output, context_idx)

                #Back propagate
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss_values.append(total_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
        self.plot_loss(loss_values)
        

    def plot_loss(self, loss_values):
        """
        Visualize the loss during training using matplotlib
        """
        plt.plot(range(1, len(loss_values) + 1), loss_values)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()


    def get_embeddings(self, words):
        """Retrieve the embeddings of words from the model"""
        word_indices = [self.word_to_idx[word] for word in words \
                        if word in self.word_to_idx]
        
        if not word_indices:  # If no known words are found, return None
            print("No known words found in the list.")
            return None
        
        word_tensor = torch.tensor(word_indices, dtype=torch.long)
        
        with torch.no_grad():  # No gradient computation needed
            embeddings = self.embeddings(word_tensor)
        
        return embeddings

                