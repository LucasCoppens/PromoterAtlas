import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from typing import List, Dict, Tuple, Optional, Union

def get_average_attention_scores(attention_scores):
    """Calculate average attention scores across all attention heads."""
    all_scores = torch.stack(attention_scores)
    avg_scores = all_scores.mean(dim=[0, 2])
    position_importance = avg_scores.sum(dim=1)
    return position_importance / position_importance.sum()

def analyse_sequence(sequence: str, model, device='cpu'):
    """Analyse a DNA sequence using the model to get logits and attention."""
    from promoter_atlas.utils.genomics import sequence_to_onehot
    
    # Convert sequence to one-hot encoding
    x = sequence_to_onehot(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, attention_scores = model(x)
        
    # Calculate position importance
    position_importance = get_average_attention_scores(attention_scores)
    
    # Create logit and attention matrices
    sequence_logit_matrix = np.zeros((len(sequence), 4))
    attention_matrix = np.zeros((len(sequence), 4))
    
    for pos, base in enumerate(sequence):
        if base in {'A', 'C', 'G', 'T'}:
            base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[base]
            if output[0, base_idx, pos].item() > 0:
                sequence_logit_matrix[pos, base_idx] = output[0, base_idx, pos].item()
            attention_matrix[pos, base_idx] = float(position_importance[0][pos])
    
    return {
        'logit_matrix': sequence_logit_matrix,
        'attention_matrix': attention_matrix
    }

def plot_sequence_analysis(sequence: str, 
                          analysis_results: Dict, 
                          title: Optional[str] = None,
                          segment_highlights: Optional[List[Tuple[int, int]]] = None,
                          segment_labels: Optional[List[str]] = None,
                          show_attention: bool = False,
                          figsize: Optional[Tuple[float, float]] = None):
    """Plot sequence logo and optionally attention map for a DNA sequence."""
    # Determine figure size
    if figsize is None:
        figsize = (8.2677, 2.35 if show_attention else 1.45)
    
    # Set up the figure and axes
    if show_attention:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})
        plt.subplots_adjust(hspace=0.5)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Define color scheme for nucleotides
    color_scheme = {
        'A': '#429747',
        'C': '#1875C7',
        'G': '#FFA500',
        'T': '#DC143C'
    }
    
    # Plot logits logo
    logo_df = pd.DataFrame(analysis_results['logit_matrix'], columns=['A', 'C', 'G', 'T'])
    logo = logomaker.Logo(logo_df, ax=ax1, color_scheme=color_scheme)
    
    # Set title if provided
    if title:
        ax1.set_title(f'{title}\n', fontsize=9)
    
    # Configure first axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    # Highlight segments if provided
    if segment_highlights:
        for i, segment in enumerate(segment_highlights):
            start = segment[0] + len(sequence) if segment[0] < 0 else segment[0]
            end = segment[1] + len(sequence) if segment[1] < 0 else segment[1]
            
            # Add the highlight
            ax1.axvspan(start - 0.5, end + 0.5, facecolor='#999999', alpha=0.23, edgecolor='none')
            
            # Add label if provided
            if segment_labels and i < len(segment_labels):
                center = (start + end) / 2
                ymin, ymax = ax1.get_ylim()
                label_y_position = ymax + 0.05 * (ymax - ymin)
                ax1.text(center, label_y_position, segment_labels[i], 
                         horizontalalignment='center', verticalalignment='bottom',
                         fontsize=8, color='black')
    
    # Handle positions for x-axis - show relative positions if analysing promoter
    positions = list(range(len(sequence)))
    step = max(len(positions) // 10, 1)
    
    # Show relative positions or absolute positions
    ax1.set_xticks(range(0, len(positions), step))
    ax1.set_xticklabels([str(i) for i in range(0, len(positions), step)])
    
    # Add attention map if requested
    if show_attention:
        ax1.set_xticks([])
        ax1.set_xlabel('')
        
        # Plot attention logo
        attention_df = pd.DataFrame(analysis_results['attention_matrix'], columns=['A', 'C', 'G', 'T'])
        attention_logo = logomaker.Logo(attention_df, ax=ax2, color_scheme=color_scheme)
        ax2.set_title('Attention Map', fontsize=8)
        ax2.set_xlabel('Position', fontsize=8)
        
        # Configure second axis
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # Use same x-ticks
        ax2.set_xticks(range(0, len(positions), step))
        ax2.set_xticklabels([str(i) for i in range(0, len(positions), step)])
    else:
        ax1.set_xlabel('Position', fontsize=8)
    
    plt.tight_layout()
    return fig