#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
from Bio import SeqIO
from BCBio import GFF
from promoter_atlas.models.promoter_segmenter import PromoterSegmenter
from promoter_atlas.annotation.genbank_annotator import annotate_genbank

def detect_format(file_path):
    """Detect file format from extension."""
    extension = Path(file_path).suffix.lower()
    
    if extension in ['.gb', '.gbk', '.genbank']:
        return 'genbank'
    elif extension in ['.gff', '.gff3']:
        return 'gff3'
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            f"Supported formats: .gb/.gbk/.genbank (GenBank) or .gff/.gff3 (GFF3). "
            f"Please rename your file with the correct extension."
        )

def load_records(file_path, file_format):
    """Load records using appropriate parser."""
    with open(file_path) as f:
        if file_format == 'genbank':
            return list(SeqIO.parse(f, "genbank"))
        elif file_format == 'gff3':
            return list(GFF.parse(f))
    return []

def write_records(records, file_path, file_format):
    """Write records using appropriate writer."""
    with open(file_path, 'w') as f:
        if file_format == 'genbank':
            SeqIO.write(records, f, "genbank")
        elif file_format == 'gff3':
            GFF.write(records, f)

def main():
    parser = argparse.ArgumentParser(description="Annotate promoter elements in genomic files")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to GenBank or GFF3 file")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save annotated file (format auto-detected from extension)")
    parser.add_argument("--model-path", type=str, 
                      default="trained_weights/segmentation/promoteratlas-annotation.pt",
                      help="Path to segmentation model weights")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Auto-detect formats
    input_format = detect_format(args.input)
    output_format = detect_format(args.output)
    print(f"Detected input format: {input_format}")
    print(f"Output format: {output_format}")
    
    # Require matching formats
    if input_format != output_format:
        raise ValueError(
            f"Input format ({input_format}) must match output format ({output_format}). "
            f"Please use the same file extension for both input and output files."
        )
    
    print(f"Loading model from {args.model_path}")
    model = PromoterSegmenter.from_pretrained(args.model_path)
    model = model.to(device)
    
    # Load input file
    print(f"Loading file: {args.input}")
    records = load_records(args.input, input_format)
    if not records:
        raise ValueError(f"No records found in {args.input}")
    
    # Process records - annotate_genbank works with any SeqRecord objects
    annotated_records = []
    total_features = 0
    
    for record in records:
        annotated_record, feature_count = annotate_genbank(record, model, device)
        annotated_records.append(annotated_record)
        total_features += feature_count
    
    # Save in requested format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_records(annotated_records, output_path, output_format)
    
    print(f"Processed {len(records)} record(s), added {total_features} features")
    print(f"Annotated file saved to: {output_path} ({output_format} format)")

if __name__ == "__main__":
    main()