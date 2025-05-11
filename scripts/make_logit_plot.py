#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import torch
from Bio import SeqIO
import matplotlib.pyplot as plt

from promoter_atlas.models.dna_transformer import DNATransformer
from promoter_atlas.utils.genomics import extract_promoter_regions
from promoter_atlas.utils.visualisation import analyse_sequence, plot_sequence_analysis

def parse_segment_highlights(segment_str):
    """Parse command line segment highlights string into list of tuples."""
    if not segment_str:
        return []
    
    segments = []
    pairs = segment_str.split(',')
    for pair in pairs:
        try:
            start, end = map(int, pair.split(':'))
            segments.append((start, end))
        except ValueError:
            print(f"Warning: Could not parse segment '{pair}'. Expected format: start:end")
    
    return segments

def main():
    parser = argparse.ArgumentParser(description='Visualise DNA sequence using the DNATransformer model')
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--fasta', type=str, help='Path to FASTA file')
    input_group.add_argument('--genbank', type=str, help='Path or accession for GenBank file')
    
    # Sequence selection options
    parser.add_argument('--locus-tag', type=str, help='Locus tag to select (required for GenBank)')
    parser.add_argument('--sequence-index', type=int, default=0, 
                      help='Index of sequence to analyse in FASTA file (default: 0)')
    
    # Model options
    parser.add_argument('--model-path', type=str, default='trained_weights/base_model/promoteratlas-base.pt',
                      help='Path to trained model weights')
    
    # Visualisation options
    parser.add_argument('--attention-map', action='store_true', help='Show attention map')
    parser.add_argument('--segment-highlights', type=str, 
                      help='Segments to highlight in format "start1:end1,start2:end2"')
    parser.add_argument('--segment-labels', type=str,
                      help='Labels for segments in format "label1,label2"')
    parser.add_argument('--output', type=str, help='Path to save the figure')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figure')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = DNATransformer()
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get sequence to analyse
    sequence = None
    title = None
    
    if args.fasta:
        # Load sequence from FASTA file
        print(f"Loading sequence from FASTA file: {args.fasta}")
        with open(args.fasta) as fasta_file:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            
        if not records:
            raise ValueError(f"No sequences found in FASTA file: {args.fasta}")
        
        if args.sequence_index >= len(records):
            raise ValueError(f"Sequence index {args.sequence_index} out of range (0-{len(records)-1})")
        
        record = records[args.sequence_index]
        sequence = str(record.seq)
        title = record.id
        
    elif args.genbank:
        # Load sequence from GenBank file
        if not args.locus_tag:
            raise ValueError("Locus tag must be specified when using GenBank input")
        
        print(f"Loading sequence from GenBank: {args.genbank}")
        # Check if it's a path or accession
        with open(args.genbank) as f:
            gbs = list(SeqIO.parse(f, "genbank"))
        
        # Extract promoter regions
        promoter_regions = extract_promoter_regions(gbs)
        
        # Find the requested locus tag
        promoter = None
        for p in promoter_regions:
            if p['locus_tag'] == args.locus_tag:
                promoter = p
                break
        
        if not promoter:
            raise ValueError(f"Locus tag {args.locus_tag} not found in GenBank file")
        
        sequence = promoter['sequence']
        if promoter['gene']:
            title = f"{promoter['organism']} {promoter['locus_tag']} ({promoter['gene']})"
        else:
            title = f"{promoter['organism']} {promoter['locus_tag']}"
    
    # Get sequence length and validate
    if not sequence:
        raise ValueError("No sequence obtained for analysis")
    
    # Analyse sequence
    print(f"Analysing sequence ({len(sequence)} bp)")
    analysis_results = analyse_sequence(sequence, model, device)
    
    # Parse segment highlights and labels
    segment_highlights = parse_segment_highlights(args.segment_highlights)
    segment_labels = args.segment_labels.split(',') if args.segment_labels else None
    
    # Create plot
    fig = plot_sequence_analysis(
        sequence=sequence,
        analysis_results=analysis_results,
        title=title,
        segment_highlights=segment_highlights,
        segment_labels=segment_labels,
        show_attention=args.attention_map
    )
    
    # Save or show plot
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    main()