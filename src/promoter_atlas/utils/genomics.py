from pathlib import Path
from Bio import SeqIO
from typing import List, Dict, Any, Optional, Union
import torch

def extract_promoter_regions(gbs, min_intergenic_length=50, regulatory_region_length=200):
    """Extract promoter regions from GenBank records with comprehensive metadata."""
    promoter_regions = []

    for gb in gbs:
        features = gb.features
        sequence = gb.seq

        genes_forward = []
        for feature in features:
            if feature.location.strand == 1 and feature.type == "CDS" and "locus_tag" in feature.qualifiers:
                gene_name = feature.qualifiers["gene"][0] if "gene" in feature.qualifiers else None
                genes_forward.append([feature.location, feature.qualifiers["locus_tag"][0], gene_name])
        
        genes_reverse = []
        for feature in features:
            if feature.location.strand == -1 and feature.type == "CDS" and "locus_tag" in feature.qualifiers:
                gene_name = feature.qualifiers["gene"][0] if "gene" in feature.qualifiers else None
                genes_reverse.append([feature.location, feature.qualifiers["locus_tag"][0], gene_name])

        # Forward strand processing
        last_end_forward = 0
        for i in range(len(genes_forward)):
            start = genes_forward[i][0].start
            end = genes_forward[i][0].end

            if start < regulatory_region_length:
                continue

            if start - last_end_forward >= min_intergenic_length:
                # Take sequence UPSTREAM of start
                prom_seq = sequence[start-regulatory_region_length:start]

                # Store promoter sequence
                promoter_regions.append({
                    'organism': gb.annotations['organism'],
                    'location': genes_forward[i][0],
                    'sequence': str(prom_seq),
                    'locus_tag': genes_forward[i][1],
                    'gene': genes_forward[i][2],
                    'strand': '+',
                    'start': start-regulatory_region_length,
                    'end': start
                })

            last_end_forward = end

        # Reverse strand processing
        last_end_reverse = len(sequence)
        genes_reverse = genes_reverse[::-1]  # Sort from end to start of genome
        for i in range(len(genes_reverse)):
            start = genes_reverse[i][0].start
            end = genes_reverse[i][0].end

            if len(sequence) - end < regulatory_region_length:
                continue
            
            if last_end_reverse - end >= min_intergenic_length:
                # Take sequence DOWNSTREAM of end (which is UPSTREAM relative to gene direction)
                prom_seq = sequence[end:end+regulatory_region_length].reverse_complement()
                # Store promoter sequence
                promoter_regions.append({
                    'organism': gb.annotations['organism'],
                    'location': genes_reverse[i][0],
                    'sequence': str(prom_seq),
                    'locus_tag': genes_reverse[i][1],
                    'gene': genes_reverse[i][2],
                    'strand': '-',
                    'start': end,
                    'end': end+regulatory_region_length
                })

            last_end_reverse = start
    
    return promoter_regions

def sequence_to_onehot(sequence: str) -> torch.Tensor:
    """Convert a DNA sequence to one-hot encoding."""
    nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = torch.zeros((4, len(sequence)))
    for i, nuc in enumerate(sequence):
        if nuc in nuc_map:  # Handle any non-standard nucleotides gracefully
            one_hot[nuc_map[nuc], i] = 1
    return one_hot