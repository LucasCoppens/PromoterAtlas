"""GFF3 file I/O utilities."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from BCBio import GFF


def load_gff3(
    gff_path: str,
    fasta_path: str
) -> tuple[List[SeqRecord], Dict[str, SeqRecord]]:
    """
    Load GFF3 annotation file with reference FASTA.

    Args:
        gff_path: Path to GFF3 file
        fasta_path: Path to reference FASTA file

    Returns:
        Tuple of (annotated_records, fasta_dict)
        - annotated_records: List of SeqRecord objects with GFF annotations
        - fasta_dict: Dictionary mapping sequence IDs to SeqRecord objects

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If files cannot be parsed
    """
    gff_file = Path(gff_path)
    fasta_file = Path(fasta_path)

    if not gff_file.exists():
        raise FileNotFoundError(f"GFF3 file not found: {gff_path}")
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Load FASTA sequences
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    if not fasta_dict:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

    # Load GFF3 and attach sequences
    with open(gff_file) as gff_handle:
        records = list(GFF.parse(gff_handle, base_dict=fasta_dict))

    if not records:
        raise ValueError(f"No records found in GFF3 file: {gff_path}")

    return records, fasta_dict


def save_gff3(
    records: List[SeqRecord],
    gff_path: str,
    include_fasta: bool = False
) -> None:
    """
    Save SeqRecord(s) to GFF3 file.

    Args:
        records: List of SeqRecord objects with features
        gff_path: Output GFF3 path
        include_fasta: Whether to include FASTA sequences at end of GFF3

    Raises:
        ValueError: If records list is empty
    """
    if not records:
        raise ValueError("Cannot save empty records list")

    path = Path(gff_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as gff_handle:
        GFF.write(records, gff_handle, include_fasta=include_fasta)


def extract_promoter_regions_gff(
    records: List[SeqRecord],
    upstream_length: int = 200,
    min_intergenic_length: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract promoter regions from GFF3-annotated records.

    Args:
        records: List of SeqRecord objects with GFF annotations
        upstream_length: Length of upstream region to extract (default: 200bp)
        min_intergenic_length: Minimum intergenic distance (default: 5bp)

    Returns:
        List of dictionaries containing promoter information:
            - organism: Organism name (from record description or 'Unknown')
            - location: BioPython FeatureLocation object
            - sequence: DNA sequence string
            - locus_tag: Locus tag or gene ID
            - gene: Gene name (if available)
            - strand: '+' or '-'
            - start: Start position
            - end: End position
            - record_id: Record ID from which region was extracted
    """
    promoter_regions = []

    for record in records:
        features = record.features
        sequence = record.seq
        organism = record.description if record.description else 'Unknown'
        record_id = record.id

        # Extract forward strand genes (CDS features)
        genes_forward = []
        for feature in features:
            if feature.type in ["CDS", "gene"] and feature.location.strand == 1:
                # Try to get locus_tag, ID, or Name
                locus_tag = (
                    feature.qualifiers.get("locus_tag", [None])[0] or
                    feature.qualifiers.get("ID", [None])[0] or
                    feature.qualifiers.get("Name", [None])[0]
                )
                if locus_tag:
                    gene_name = feature.qualifiers.get("Name", [None])[0]
                    genes_forward.append([
                        feature.location,
                        locus_tag,
                        gene_name
                    ])

        # Extract reverse strand genes
        genes_reverse = []
        for feature in features:
            if feature.type in ["CDS", "gene"] and feature.location.strand == -1:
                locus_tag = (
                    feature.qualifiers.get("locus_tag", [None])[0] or
                    feature.qualifiers.get("ID", [None])[0] or
                    feature.qualifiers.get("Name", [None])[0]
                )
                if locus_tag:
                    gene_name = feature.qualifiers.get("Name", [None])[0]
                    genes_reverse.append([
                        feature.location,
                        locus_tag,
                        gene_name
                    ])

        # Process forward strand
        last_end_forward = 0
        for location, locus_tag, gene_name in genes_forward:
            start = int(location.start)
            end = int(location.end)

            # Skip if too close to sequence start
            if start < upstream_length:
                continue

            # Check intergenic distance
            if start - last_end_forward >= min_intergenic_length:
                # Extract upstream sequence
                prom_seq = sequence[start - upstream_length:start]

                promoter_regions.append({
                    'organism': organism,
                    'location': location,
                    'sequence': str(prom_seq),
                    'locus_tag': locus_tag,
                    'gene': gene_name,
                    'strand': '+',
                    'start': start - upstream_length,
                    'end': start,
                    'record_id': record_id
                })

            last_end_forward = end

        # Process reverse strand (sorted from end to start)
        last_end_reverse = len(sequence)
        genes_reverse = genes_reverse[::-1]

        for location, locus_tag, gene_name in genes_reverse:
            start = int(location.start)
            end = int(location.end)

            # Skip if too close to sequence end
            if len(sequence) - end < upstream_length:
                continue

            # Check intergenic distance
            if last_end_reverse - end >= min_intergenic_length:
                # Extract downstream sequence and reverse complement
                prom_seq = sequence[end:end + upstream_length].reverse_complement()

                promoter_regions.append({
                    'organism': organism,
                    'location': location,
                    'sequence': str(prom_seq),
                    'locus_tag': locus_tag,
                    'gene': gene_name,
                    'strand': '-',
                    'start': end,
                    'end': end + upstream_length,
                    'record_id': record_id
                })

            last_end_reverse = start

    return promoter_regions
