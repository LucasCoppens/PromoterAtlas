"""I/O utilities for different genome annotation formats."""

from .formats import detect_format, validate_format
from .genbank_io import (
    load_genbank,
    save_genbank,
    extract_promoter_regions_genbank
)
from .gff_io import (
    load_gff3,
    save_gff3,
    extract_promoter_regions_gff
)
from .fasta_io import (
    load_fasta,
    validate_fasta_lengths,
    create_sliding_windows,
    prepare_fasta_for_annotation,
    save_fasta_results_tsv,
    save_fasta_results_json,
)

__all__ = [
    'detect_format',
    'validate_format',
    'load_genbank',
    'save_genbank',
    'extract_promoter_regions_genbank',
    'load_gff3',
    'save_gff3',
    'extract_promoter_regions_gff',
    'load_fasta',
    'validate_fasta_lengths',
    'create_sliding_windows',
    'prepare_fasta_for_annotation',
    'save_fasta_results_tsv',
    'save_fasta_results_json',
]
