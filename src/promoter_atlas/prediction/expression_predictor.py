import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from promoter_atlas.utils.genomics import sequence_to_onehot

class ExpressionPrediction:
    """Class for predicting gene expression from DNA sequences."""
    
    def __init__(self, model, device='cpu', normalization_params=None):
        """Initialize with a model and optional normalization parameters."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.normalization_params = normalization_params or {}
    
    @classmethod
    def load(cls, model_path, device=None):
        """Load a predictor with the specified model."""
        from promoter_atlas.models.expression_predictor import ExpressionPredictor
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load the model
        model = ExpressionPredictor.from_pretrained(model_path)
        
        # Try to load normalization parameters from the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'normalization_params' in checkpoint:
            norm_params = checkpoint['normalization_params']
        
        return cls(model, device, norm_params)
    
    def predict(self, sequences: List[str]) -> np.ndarray:
        """Predict expression levels for a list of sequences."""
        # Process sequences in batches to avoid memory issues
        batch_size = 32
        predictions = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_tensors = [sequence_to_onehot(seq) for seq in batch_seqs]
            
            # Pad sequences to the same length if necessary
            max_len = max(tensor.shape[1] for tensor in batch_tensors)
            padded_tensors = []
            
            for tensor in batch_tensors:
                if tensor.shape[1] < max_len:
                    padding = torch.zeros((4, max_len - tensor.shape[1]))
                    padded = torch.cat([padding, tensor], dim=1)
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)
            
            # Stack into batch
            batch_input = torch.stack(padded_tensors).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                batch_preds = self.model(batch_input)
                predictions.append(batch_preds.cpu().numpy())
        
        # Combine predictions
        predictions = np.vstack(predictions).squeeze()
        
        # Denormalize if parameters are available
        if self.normalization_params:
            predictions = self._denormalize_predictions(predictions)
            
        return predictions
    
    def _denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        if not self.normalization_params:
            return predictions
            
        # Undo normalization
        mean = self.normalization_params.get('mean', 0)
        std = self.normalization_params.get('std', 1)
        denorm = predictions * std + mean
        
        # Undo log transform if applied
        if self.normalization_params.get('log_transform', False):
            denorm = np.exp(denorm) - 1
            
        return denorm
    
    def predict_fasta(self, fasta_path: Union[str, Path]) -> Dict[str, float]:
        """Predict expression levels for sequences in a FASTA file."""
        from Bio import SeqIO
        
        # Load sequences from FASTA
        sequences = []
        identifiers = []
        
        with open(fasta_path) as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences.append(str(record.seq))
                identifiers.append(record.id)
        
        # Make predictions
        predictions = self.predict(sequences)
        
        # Return dictionary mapping sequence IDs to predicted values
        return dict(zip(identifiers, predictions))