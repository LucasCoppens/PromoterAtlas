"""FASTA-specific annotation utilities."""

from collections import defaultdict
from typing import List, Dict, Any

import torch

from promoter_atlas.utils.genomics import sequence_to_onehot
from promoter_atlas.annotation.annotator import (
    find_consecutive_segments,
    LABEL_MAP
)


def annotate_fasta_windows(
    windows: List[Dict[str, Any]],
    model,
    device: str = 'cpu',
    min_segment_length: int = 3,
    apply_cooccurrence: bool = True
) -> List[Dict[str, Any]]:
    """
    Annotate FASTA sequence windows with regulatory elements.

    Args:
        windows: List of window dictionaries from create_sliding_windows
        model: PromoterSegmenter model
        device: Device to run inference on
        min_segment_length: Minimum consecutive bases for regulatory segments
        apply_cooccurrence: Apply co-occurrence rules for paired elements

    Returns:
        List of result dictionaries, each containing:
            - seq_id: Sequence identifier
            - window_start: Window start in parent sequence
            - window_end: Window end in parent sequence
            - elements: List of detected elements, each with:
                - type: Element type name
                - label: Numeric label
                - start: Start position within window
                - end: End position within window
                - abs_start: Absolute start in parent sequence
                - abs_end: Absolute end in parent sequence
    """
    results = []
    model.eval()

    for window in windows:
        # Convert sequence to one-hot encoding
        x = sequence_to_onehot(window['sequence']).unsqueeze(0).to(device)

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

        # Build result entry
        elements = []
        for segment in segments:
            if segment['label'] in LABEL_MAP:
                elements.append({
                    'type': LABEL_MAP[segment['label']],
                    'label': segment['label'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'abs_start': window['window_start'] + segment['start'],
                    'abs_end': window['window_start'] + segment['end']
                })

        results.append({
            'seq_id': window['seq_id'],
            'window_start': window['window_start'],
            'window_end': window['window_end'],
            'elements': elements
        })

    return results


def merge_overlapping_predictions(
    results: List[Dict[str, Any]],
    window_size: int = 200
) -> List[Dict[str, Any]]:
    """
    Merge overlapping predictions from sliding windows using consensus.

    For each position, uses the prediction from the window where that
    position is most central (furthest from window edges).

    Args:
        results: List of result dictionaries from annotate_fasta_windows
        window_size: Window size used for annotation

    Returns:
        List of merged result dictionaries per sequence
    """
    # Group results by sequence
    by_sequence = defaultdict(list)
    for result in results:
        by_sequence[result['seq_id']].append(result)

    merged_results = []

    for seq_id, seq_results in by_sequence.items():
        # Find the total sequence length
        max_pos = max(r['window_end'] for r in seq_results)

        # For each position, track predictions with their "centrality" score
        # Centrality = distance from nearest window edge
        position_predictions = defaultdict(list)

        for result in seq_results:
            window_start = result['window_start']

            for element in result['elements']:
                for pos in range(element['abs_start'], element['abs_end']):
                    # Centrality score: how far from window edges
                    dist_from_start = pos - window_start
                    dist_from_end = (window_start + window_size) - pos
                    centrality = min(dist_from_start, dist_from_end)

                    position_predictions[pos].append({
                        'label': element['label'],
                        'type': element['type'],
                        'centrality': centrality
                    })

        # For each position, pick prediction with highest centrality
        final_predictions = {}
        for pos, preds in position_predictions.items():
            best = max(preds, key=lambda x: x['centrality'])
            final_predictions[pos] = best['label']

        # Convert back to segments
        if final_predictions:
            positions = sorted(final_predictions.keys())
            segments = []
            current_label = final_predictions[positions[0]]
            current_start = positions[0]

            for i in range(1, len(positions)):
                pos = positions[i]
                label = final_predictions[pos]

                # Check for gap or label change
                if pos != positions[i - 1] + 1 or label != current_label:
                    if current_label != 0:
                        segments.append({
                            'type': LABEL_MAP.get(current_label, f"Unknown-{current_label}"),
                            'label': current_label,
                            'abs_start': current_start,
                            'abs_end': positions[i - 1] + 1
                        })
                    current_label = label
                    current_start = pos

            # Don't forget last segment
            if current_label != 0:
                segments.append({
                    'type': LABEL_MAP.get(current_label, f"Unknown-{current_label}"),
                    'label': current_label,
                    'abs_start': current_start,
                    'abs_end': positions[-1] + 1
                })

            merged_results.append({
                'seq_id': seq_id,
                'sequence_length': max_pos,
                'elements': segments
            })

    return merged_results
