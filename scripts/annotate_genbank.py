#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
from Bio import SeqIO

from promoter_atlas.models.promoter_segmenter import PromoterSegmenter
from promoter_atlas.annotation.genbank_annotator import annotate_genbank

def main():
    parser = argparse.ArgumentParser(description="Annotate promoter elements in a GenBank file")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to GenBank file or accession number")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save annotated GenBank file")
    parser.add_argument("--model-path", type=str, 
                      default="trained_weights/segmentation/promoteratlas-annotation.pt",
                      help="Path to segmentation model weights")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = PromoterSegmenter.from_pretrained(args.model_path)
    model = model.to(device)
    
    # Load GenBank file or accession
    input_path = Path(args.input)
    print(f"Loading GenBank file: {input_path}")
    with open(input_path) as f:
        records = list(SeqIO.parse(f, "genbank"))

    if not records:
        raise ValueError(f"No records found in {args.input}")
    
    # Process each record
    annotated_records = []
    total_features = 0
    
    for record in records:
        annotated_record, feature_count = annotate_genbank(
            record, model, device)
        annotated_records.append(annotated_record)
        total_features += feature_count
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        SeqIO.write(annotated_records, f, "genbank")
    
    print(f"Processed {len(records)} record(s), added {total_features} features")
    print(f"Annotated GenBank saved to: {output_path}")

if __name__ == "__main__":
    main()