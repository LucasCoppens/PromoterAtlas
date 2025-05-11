from Bio.SeqFeature import SeqFeature, FeatureLocation
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from pathlib import Path
from promoter_atlas.utils.genomics import extract_promoter_regions, sequence_to_onehot

# Mapping of numeric labels to regulatory elements
LABEL_MAP = {
    1: "RBS",
    2: "σ70 / σ38 promoter -10 element",
    3: "σ70 / σ38 promoter -35 element",
    4: "σ54 promoter -12 element",
    5: "σ54 promoter -24 element",
    6: "σ32 promoter -10 element",
    7: "σ32 promoter -35 element",
    8: "σ28 promoter -10 element",
    9: "σ28 promoter -35 element",
    10: "σ24 promoter -10 element",
    11: "σ24 promoter -35 element",
}

def find_consecutive_segments(predictions, min_length=4):
    """Find segments with at least min_length consecutive same predictions."""
    segments = []
    current_label = predictions[0]
    current_start = 0
    current_length = 1
    
    for i in range(1, len(predictions)):
        if predictions[i] == current_label:
            current_length += 1
        else:
            if current_length >= min_length and current_label != 0:  # Ignore label 0
                segments.append({
                    'label': current_label,
                    'start': current_start,
                    'end': i
                })
            current_label = predictions[i]
            current_start = i
            current_length = 1
    
    # Check last segment
    if current_length >= min_length and current_label != 0:
        segments.append({
            'label': current_label,
            'start': current_start,
            'end': len(predictions)
        })
    
    # Apply co-occurrence rules
    return apply_cooccurrence_rules(segments)

def apply_cooccurrence_rules(segments):
    """Apply co-occurrence rules for promoter element pairs."""
    # Define pairs that must co-occur
    required_pairs = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    
    # Get existing labels
    existing_labels = set(segment['label'] for segment in segments)
    
    # Identify which segments to keep
    segments_to_keep = []
    
    for segment in segments:
        label = segment['label']
        keep_segment = True
        
        # Check if this label is part of a required pair
        for pair in required_pairs:
            if label in pair:
                # Find the partner label
                partner_label = pair[1] if label == pair[0] else pair[0]
                
                # If partner doesn't exist, mark for removal
                if partner_label not in existing_labels:
                    keep_segment = False
                    break
        
        if keep_segment:
            segments_to_keep.append(segment)
    
    return segments_to_keep

def create_regulatory_feature(segment, promoter):
    """Create a SeqFeature for a regulatory element."""
    label = segment['label']
    if label not in LABEL_MAP:
        return None
        
    if promoter['strand'] == '+':
        feature_start = promoter['start'] + segment['start']
        feature_end = promoter['start'] + segment['end']
        strand = 1
    else:
        # For reverse strand, we need to count from the end
        feature_start = promoter['end'] - segment['end']
        feature_end = promoter['end'] - segment['start']
        strand = -1
    
    feature = SeqFeature(
        FeatureLocation(feature_start, feature_end, strand=strand),
        type="regulatory",
        qualifiers={
            "regulatory_class": LABEL_MAP[label],
            "note": f"Predicted by PromoterAtlas segmentation model for gene {promoter['locus_tag']}"
        }
    )
    
    return feature

def annotate_genbank(gb_record, model, device='cpu', min_segment_length=4):
    """Annotate a GenBank record with promoter elements."""
    print(f"Extracting promoter regions from {gb_record.id}")
    promoter_regions = extract_promoter_regions([gb_record])
    print(f"Found {len(promoter_regions)} promoter regions")
    
    if not promoter_regions:
        print(f"No promoter regions found in {gb_record.id}")
        return gb_record, 0
    
    feature_count = 0
    model.eval()
    
    # Process each promoter region
    for promoter in promoter_regions:
        # Convert sequence to one-hot encoding
        x = sequence_to_onehot(promoter['sequence']).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(x)
            predictions = logits.argmax(dim=1)[0].cpu().tolist()
        
        # Find consecutive segment predictions
        segments = find_consecutive_segments(predictions, min_length=min_segment_length)
        
        # Create and add features
        for segment in segments:
            feature = create_regulatory_feature(segment, promoter)
            if feature:
                gb_record.features.append(feature)
                feature_count += 1
    
    # Sort features by position
    gb_record.features.sort(key=lambda x: x.location.start)
    print(f"Added {feature_count} regulatory features to {gb_record.id}")
    
    return gb_record, feature_count