"""Genome annotation utilities."""

from .genbank_annotator import annotate_genbank
from .annotator import annotate_records, LABEL_MAP

__all__ = [
    'annotate_genbank',
    'annotate_records',
    'LABEL_MAP',
]
