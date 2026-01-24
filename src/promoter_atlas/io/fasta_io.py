"""FASTA file I/O utilities for direct sequence annotation."""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def load_fasta(filepath: str) -> List[SeqRecord]:
    """
    Load FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        List of SeqRecord objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no records found
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")

    with open(path) as f:
        records = list(SeqIO.parse(f, "fasta"))

    if not records:
        raise ValueError(f"No FASTA records found in {filepath}")

    return records


def validate_fasta_lengths(
    records: List[SeqRecord],
    min_length: int = 200
) -> None:
    """
    Validate that all sequences meet minimum length requirement.

    Args:
        records: List of SeqRecord objects
        min_length: Minimum required sequence length (default: 200)

    Raises:
        ValueError: If any sequence is shorter than min_length
    """
    short_sequences = []
    for record in records:
        if len(record.seq) < min_length:
            short_sequences.append((record.id, len(record.seq)))

    if short_sequences:
        msg = f"Sequences must be at least {min_length}bp. Found {len(short_sequences)} sequence(s) too short:\n"
        for seq_id, length in short_sequences[:5]:
            msg += f"  - {seq_id}: {length}bp\n"
        if len(short_sequences) > 5:
            msg += f"  ... and {len(short_sequences) - 5} more\n"
        raise ValueError(msg)


def create_sliding_windows(
    sequence: str,
    seq_id: str,
    window_size: int = 200,
    step_size: int = 20
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Create sliding windows for a sequence.

    Args:
        sequence: DNA sequence string
        seq_id: Sequence identifier
        window_size: Size of each window (default: 200, matches model)
        step_size: Step size between windows (default: 20)

    Returns:
        Tuple of (list of window dicts, needs_sliding_window)
        Each window dict contains:
            - sequence: Window sequence
            - seq_id: Parent sequence ID
            - window_start: Start position in parent sequence (0-indexed)
            - window_end: End position in parent sequence
    """
    seq_len = len(sequence)
    windows = []

    # If sequence is exactly window_size, just one window
    if seq_len == window_size:
        windows.append({
            'sequence': sequence,
            'seq_id': seq_id,
            'window_start': 0,
            'window_end': window_size
        })
        return windows, False

    # Sliding window needed
    needs_sliding = True

    # Generate windows with step_size
    pos = 0
    while pos + window_size <= seq_len:
        windows.append({
            'sequence': sequence[pos:pos + window_size],
            'seq_id': seq_id,
            'window_start': pos,
            'window_end': pos + window_size
        })
        pos += step_size

    # Always include the last frame (final window_size bp)
    last_start = seq_len - window_size
    if windows[-1]['window_start'] != last_start:
        windows.append({
            'sequence': sequence[last_start:seq_len],
            'seq_id': seq_id,
            'window_start': last_start,
            'window_end': seq_len
        })

    return windows, needs_sliding


def prepare_fasta_for_annotation(
    records: List[SeqRecord],
    window_size: int = 200,
    step_size: int = 20
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Prepare FASTA sequences for annotation using sliding windows.

    Args:
        records: List of SeqRecord objects
        window_size: Size of each window (default: 200)
        step_size: Step size between windows (default: 20)

    Returns:
        Tuple of (list of all windows, count of sequences needing sliding window)
    """
    all_windows = []
    sliding_count = 0

    for record in records:
        windows, needs_sliding = create_sliding_windows(
            str(record.seq).upper(),
            record.id,
            window_size=window_size,
            step_size=step_size
        )
        all_windows.extend(windows)
        if needs_sliding:
            sliding_count += 1

    return all_windows, sliding_count


def save_fasta_results_tsv(
    results: List[Dict[str, Any]],
    filepath: str,
    merged: bool = False
) -> None:
    """
    Save FASTA annotation results to TSV file.

    Args:
        results: List of result dictionaries
        filepath: Output TSV path
        merged: Whether results are merged (affects column names)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        if merged:
            # Merged format: no window information
            writer.writerow([
                'sequence_id', 'element_type', 'start', 'end', 'length'
            ])

            for result in results:
                for element in result['elements']:
                    writer.writerow([
                        result['seq_id'],
                        element['type'],
                        element['abs_start'],
                        element['abs_end'],
                        element['abs_end'] - element['abs_start']
                    ])
        else:
            # Windowed format: includes window context
            writer.writerow([
                'sequence_id', 'window_start', 'window_end',
                'element_type', 'element_start', 'element_end',
                'abs_start', 'abs_end'
            ])

            for result in results:
                for element in result['elements']:
                    writer.writerow([
                        result['seq_id'],
                        result['window_start'],
                        result['window_end'],
                        element['type'],
                        element['start'],
                        element['end'],
                        element['abs_start'],
                        element['abs_end']
                    ])


def save_fasta_results_json(
    results: List[Dict[str, Any]],
    filepath: str
) -> None:
    """
    Save FASTA annotation results to JSON file.

    Args:
        results: List of result dictionaries
        filepath: Output JSON path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
