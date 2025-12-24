"""File format detection and validation utilities."""

from pathlib import Path
from typing import Optional


def detect_format(filepath: str) -> str:
    """
    Detect file format from file extension.

    Args:
        filepath: Path to the file

    Returns:
        Format string: 'genbank', 'gff3', or 'fasta'

    Raises:
        ValueError: If format cannot be determined
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    # GenBank formats
    if suffix in ['.gb', '.gbk', '.genbank']:
        return 'genbank'

    # GFF formats
    if suffix in ['.gff', '.gff3']:
        return 'gff3'

    # FASTA formats
    if suffix in ['.fa', '.fasta', '.fna', '.ffn']:
        return 'fasta'

    raise ValueError(
        f"Cannot detect format from file extension '{suffix}'. "
        f"Supported extensions: .gb, .gbk, .genbank (GenBank), "
        f".gff, .gff3 (GFF3), .fa, .fasta, .fna, .ffn (FASTA)"
    )


def validate_format(filepath: str, expected_format: str) -> bool:
    """
    Validate that a file matches the expected format.

    Args:
        filepath: Path to the file
        expected_format: Expected format ('genbank', 'gff3', or 'fasta')

    Returns:
        True if format matches, False otherwise
    """
    try:
        detected = detect_format(filepath)
        return detected == expected_format
    except ValueError:
        return False
