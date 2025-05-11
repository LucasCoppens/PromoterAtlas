import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class PromoterSegmenter(nn.Module):
    """Neural network for segmenting promoter regions into functional elements."""
    
    def __init__(self, base_model=None, base_model_path=None):
        super().__init__()
        
        # Base model specification
        if base_model is None:
            from promoter_atlas.models.dna_transformer import DNATransformer
            base_model = DNATransformer()
            
            # Load base model weights if provided
            if base_model_path is not None:
                if torch.cuda.is_available():
                    checkpoint = torch.load(base_model_path)
                else:
                    checkpoint = torch.load(base_model_path, map_location=torch.device('cpu'))
                                        
                base_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.base_model = base_model
        
        # Get dimension of base model
        base_dim = self.base_model.blocks[0].feed_forward_up.in_features  # 128
            
        # Freeze base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Additional layers for segmentation
        self.conv1 = nn.Conv1d(base_dim, base_dim, kernel_size=10, stride=1, padding='same')
        self.conv2 = nn.Conv1d(base_dim, base_dim, kernel_size=10, stride=1, padding='same')
        self.out_conv = nn.Conv1d(base_dim, 12, kernel_size=10, stride=1, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        """Process input sequence through model to get segmentation logits."""
        # Get embeddings from last layer of base model
        with torch.no_grad():
            _, _, latent = self.base_model(x, return_latent=True)
        
        # Pass through additional layers
        x = latent + self.relu(self.conv1(latent))
        x = x + self.relu(self.conv2(x))
        logits = self.out_conv(x)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, model_path=None):
        """Load a pretrained PromoterSegmenter model."""
        if model_path is None:
            model_path = Path("trained_weights/segmentation/promoteratlas-annotation.pt")
            
        # Create model instance
        model = cls()
        
        # Load weights
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
        model.load_state_dict(checkpoint)
        return model