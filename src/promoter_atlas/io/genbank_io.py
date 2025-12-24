"""GenBank file I/O utilities."""

from pathlib import Path
from typing import List, Dict, Any
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def load_genbank(filepath: str) -> List[SeqRecord]:
    """
    Load GenBank file(s).

    Args:
        filepath: Path to GenBank file

    Returns:
        List of SeqRecord objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no records found or parsing fails
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GenBank file not found: {filepath}")

    with open(path) as f:
        records = list(SeqIO.parse(f, "genbank"))

    if not records:
        raise ValueError(f"No GenBank records found in {filepath}")

    return records


def save_genbank(records: List[SeqRecord], filepath: str) -> None:
    """
    Save SeqRecord(s) to GenBank file.

    Args:
        records: List of SeqRecord objects
        filepath: Output path

    Raises:
        ValueError: If records list is empty
    """
    if not records:
        raise ValueError("Cannot save empty records list")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        SeqIO.write(records, f, "genbank")


def extract_promoter_regions_genbank(
    records: List[SeqRecord],
    upstream_length: int = 200,
    min_intergenic_length: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract promoter regions from GenBank records.

    Args:
        records: List of GenBank SeqRecord objects
        upstream_length: Length of upstream region to extract (default: 200bp)
        min_intergenic_length: Minimum intergenic distance (default: 5bp)

    Returns:
        List of dictionaries containing promoter information:
            - organism: Organism name
            - location: BioPython FeatureLocation object
            - sequence: DNA sequence string
            - locus_tag: Locus tag
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
        organism = record.annotations.get('organism', 'Unknown')
        record_id = record.id

        # Extract forward strand genes
        genes_forward = []
        for feature in features:
            if (feature.location.strand == 1 and
                feature.type == "CDS" and
                "locus_tag" in feature.qualifiers):
                gene_name = feature.qualifiers.get("gene", [None])[0]
                genes_forward.append([
                    feature.location,
                    feature.qualifiers["locus_tag"][0],
                    gene_name
                ])

        # Extract reverse strand genes
        genes_reverse = []
        for feature in features:
            if (feature.location.strand == -1 and
                feature.type == "CDS" and
                "locus_tag" in feature.qualifiers):
                gene_name = feature.qualifiers.get("gene", [None])[0]
                genes_reverse.append([
                    feature.location,
                    feature.qualifiers["locus_tag"][0],
                    gene_name
                ])

        # Process forward strand
        last_end_forward = 0
        for location, locus_tag, gene_name in genes_forward:
            start = location.start
            end = location.end

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
            start = location.start
            end = location.end

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
