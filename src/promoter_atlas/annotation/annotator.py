"""Format-agnostic genome annotation utilities."""

from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
from typing import List, Dict, Any
import torch
from promoter_atlas.utils.genomics import sequence_to_onehot

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


def find_consecutive_segments(
    predictions: List[int],
    min_length: int = 3,
    apply_cooccurrence: bool = True
) -> List[Dict[str, Any]]:
    """
    Find segments with at least min_length consecutive same predictions.

    Args:
        predictions: List of predicted class labels
        min_length: Minimum consecutive length for a valid segment (default: 3)
        apply_cooccurrence: Whether to apply co-occurrence rules (default: True)

    Returns:
        List of segment dictionaries with 'label', 'start', 'end'
    """
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

    # Apply co-occurrence rules if requested
    if apply_cooccurrence:
        return apply_cooccurrence_rules(segments)
    else:
        return segments


def apply_cooccurrence_rules(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply co-occurrence rules for promoter element pairs.

    Certain promoter elements must appear together (e.g., -10 and -35 elements).
    This function filters segments to ensure paired elements both exist.

    Args:
        segments: List of segment dictionaries

    Returns:
        Filtered list of segments where paired elements both exist
    """
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


def create_regulatory_feature(
    segment: Dict[str, Any],
    promoter: Dict[str, Any]
) -> SeqFeature:
    """
    Create a SeqFeature for a regulatory element.

    Args:
        segment: Dictionary with 'label', 'start', 'end'
        promoter: Dictionary with promoter metadata

    Returns:
        BioPython SeqFeature object, or None if invalid
    """
    label = segment['label']
    if label not in LABEL_MAP:
        return None

    if promoter['strand'] == '+':
        feature_start = promoter['start'] + segment['start']
        feature_end = promoter['start'] + segment['end']
        strand = 1
    else:
        # For reverse strand, count from the end
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


def annotate_records(
    records: List[SeqRecord],
    promoter_regions: List[Dict[str, Any]],
    model,
    device: str = 'cpu',
    min_segment_length: int = 3,
    apply_cooccurrence: bool = True
) -> tuple[List[SeqRecord], int]:
    """
    Annotate genome records with promoter elements.

    This function is format-agnostic and works with any SeqRecord objects.

    Args:
        records: List of SeqRecord objects to annotate
        promoter_regions: List of promoter region dictionaries from extraction
        model: PromoterSegmenter model
        device: Device to run inference on ('cpu' or 'cuda')
        min_segment_length: Minimum consecutive bases for regulatory segments (default: 3)
        apply_cooccurrence: Apply co-occurrence rules for paired elements (default: True)

    Returns:
        Tuple of (annotated_records, feature_count)
    """
    print(f"Annotating {len(records)} record(s) with {len(promoter_regions)} promoter regions")

    if not promoter_regions:
        print("No promoter regions found")
        return records, 0

    feature_count = 0
    model.eval()

    # Create a mapping of record_id to record for efficient lookup
    record_map = {record.id: record for record in records}

    # Process each promoter region
    for promoter in promoter_regions:
        # Convert sequence to one-hot encoding
        x = sequence_to_onehot(promoter['sequence']).unsqueeze(0).to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(x)
            predictions = logits.argmax(dim=1)[0].cpu().tolist()

        # Find consecutive segment predictions
        segments = find_consecutive_segments(
            predictions,
            min_length=min_segment_length,
            apply_cooccurrence=apply_cooccurrence
        )

        # Create and add features to the corresponding record
        record_id = promoter.get('record_id', records[0].id)
        if record_id in record_map:
            record = record_map[record_id]

            for segment in segments:
                feature = create_regulatory_feature(segment, promoter)
                if feature:
                    record.features.append(feature)
                    feature_count += 1

    # Sort features by position for each record
    for record in records:
        record.features.sort(key=lambda x: x.location.start)

    print(f"Added {feature_count} regulatory features")

    return records, feature_count
