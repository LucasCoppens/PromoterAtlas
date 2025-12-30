#!/usr/bin/env python
"""
Unified genome annotation script supporting multiple file formats.

Supports:
- GenBank (.gb, .gbk, .genbank)
- GFF3 (.gff, .gff3) with reference FASTA

Annotates bacterial genomes with promoter regulatory elements using PromoterAtlas.
"""

import argparse
from pathlib import Path
import torch

from promoter_atlas.models.promoter_segmenter import PromoterSegmenter
from promoter_atlas.annotation.annotator import annotate_records
from promoter_atlas.io import (
    detect_format,
    load_genbank,
    save_genbank,
    load_gff3,
    save_gff3,
    extract_promoter_regions_genbank,
    extract_promoter_regions_gff
)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate promoter elements in bacterial genomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate GenBank file
  python annotate.py --input genome.gb --output annotated.gb

  # Annotate GFF3 file (requires reference FASTA)
  python annotate.py --input genome.gff3 --reference genome.fasta --output annotated.gff3

  # Specify format explicitly
  python annotate.py --input genome.gff --reference genome.fna --output annotated.gff --format gff3

  # Customize upstream region length
  python annotate.py --input genome.gb --output annotated.gb --upstream-length 300
"""
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input genome file (GenBank or GFF3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save annotated genome file"
    )

    # Format arguments
    parser.add_argument(
        "--format",
        choices=["genbank", "gff3", "auto"],
        default="auto",
        help="Input file format (default: auto-detect from extension)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Reference FASTA file (required for GFF3 input)"
    )
    parser.add_argument(
        "--output-format",
        choices=["genbank", "gff3", "auto"],
        default="auto",
        help="Output file format (default: same as input)"
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_weights/segmentation/promoteratlas-annotation.pt",
        help="Path to PromoterAtlas segmentation model weights"
    )

    # Annotation parameters
    parser.add_argument(
        "--upstream-length",
        type=int,
        default=200,
        help="Length of upstream region to extract (bp, default: 200)"
    )
    parser.add_argument(
        "--min-intergenic-length",
        type=int,
        default=5,
        help="Minimum intergenic distance (bp, default: 5)"
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=3,
        help="Minimum consecutive bases for regulatory segments (bp, default: 3)"
    )
    parser.add_argument(
        "--no-cooccurrence",
        action="store_true",
        help="Disable co-occurrence rules for paired promoter elements"
    )

    # Device argument
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for inference (default: auto-detect)"
    )

    args = parser.parse_args()

    # Detect input format
    if args.format == "auto":
        try:
            input_format = detect_format(args.input)
            print(f"Detected input format: {input_format}")
        except ValueError as e:
            parser.error(str(e))
    else:
        input_format = args.format

    # Validate GFF3 requirements
    if input_format == "gff3" and not args.reference:
        parser.error("--reference FASTA file is required when using GFF3 input")

    # Determine output format
    if args.output_format == "auto":
        output_format = input_format
    else:
        output_format = args.output_format

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading PromoterAtlas model from {args.model_path}")
    model = PromoterSegmenter.from_pretrained(args.model_path)
    model = model.to(device)

    # Load genome based on format
    print(f"Loading {input_format.upper()} file: {args.input}")
    if input_format == "genbank":
        records = load_genbank(args.input)
        promoter_regions = extract_promoter_regions_genbank(
            records,
            upstream_length=args.upstream_length,
            min_intergenic_length=args.min_intergenic_length
        )
    elif input_format == "gff3":
        records, _ = load_gff3(args.input, args.reference)
        promoter_regions = extract_promoter_regions_gff(
            records,
            upstream_length=args.upstream_length,
            min_intergenic_length=args.min_intergenic_length
        )
    else:
        parser.error(f"Unsupported format: {input_format}")

    print(f"Loaded {len(records)} record(s)")
    print(f"Extracted {len(promoter_regions)} promoter regions")

    if not promoter_regions:
        print("WARNING: No promoter regions found. Check your input file has CDS features.")
        print("Saving original records without annotations.")
    else:
        # Annotate records
        records, feature_count = annotate_records(
            records,
            promoter_regions,
            model,
            device=str(device),
            min_segment_length=args.min_segment_length,
            apply_cooccurrence=not args.no_cooccurrence
        )
        print(f"Added {feature_count} regulatory features")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving annotated {output_format.upper()} to: {output_path}")
    if output_format == "genbank":
        save_genbank(records, args.output)
    elif output_format == "gff3":
        save_gff3(records, args.output, include_fasta=False)
    else:
        parser.error(f"Unsupported output format: {output_format}")

    print("Done!")


if __name__ == "__main__":
    main()
