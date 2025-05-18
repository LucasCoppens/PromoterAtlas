import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Dict

class SequenceDataset(Dataset):
    def __init__(self, h5_file: str):
        """Initialize the dataset from an HDF5 file containing DNA sequences."""
        self.h5_file = h5_file
        
        # Keep file handle open for faster access
        self.f = h5py.File(self.h5_file, 'r')
        self.sequences = self.f['sequences']
        self.total_sequences = self.f.attrs['total_sequences']
        
        # Create shuffled indices for the entire dataset
        self.indices = np.random.permutation(self.total_sequences)
    
    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence from the dataset."""
        real_idx = self.indices[idx]
        orig_seq = self.sequences[real_idx]  # Expected shape: (sequence_length, 4)
        
        # Permute to (4, sequence_length) and ensure contiguous memory layout
        return {
            'sequence': torch.from_numpy(orig_seq).permute(1, 0).contiguous()
        }
    
    def __del__(self):
        """Ensure HDF5 file is properly closed"""
        try:
            if hasattr(self, 'f'):
                self.f.close()
        except:
            pass