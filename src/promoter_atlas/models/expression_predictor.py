import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from promoter_atlas.models.dna_transformer import DNATransformer as BaseArchitecture

class ExpressionPredictor(nn.Module):
    """Neural network for predicting gene expression levels from DNA sequences."""
        
    def __init__(self, seq_length=86):
        super().__init__()
        
        # Load base model from fixed path
        base_model_path = Path("trained_weights/base_model/promoteratlas-base.pt")
        self.base_model = BaseArchitecture()

        if torch.cuda.is_available():
            checkpoint = torch.load(base_model_path)
        else:
            checkpoint = torch.load(base_model_path, map_location=torch.device('cpu'))
                                    
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get dimension of base model
        base_dim = self.base_model.blocks[0].feed_forward_up.in_features  # 128
        
        # After flattening we have base_dim * seq_length
        fc_input_dim = base_dim * seq_length
            
        # Freeze base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Additional layers
        self.conv1 = nn.Conv1d(base_dim, base_dim, kernel_size=8, stride=1, padding='same')
        self.conv2 = nn.Conv1d(base_dim, base_dim, kernel_size=8, stride=1, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=7, stride=1, padding=3)
        self.fc1 = nn.Linear(fc_input_dim, 64)  # Input dim matches flattened size
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Get embeddings from last layer of base model
        with torch.no_grad():
            _, _, latent = self.base_model(x, return_latent=True)
        
        # Pass through additional layers
        x = self.relu(self.conv1(latent))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)  # Flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load a pretrained expression prediction model."""
        # Create model instance
        model = cls()
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model