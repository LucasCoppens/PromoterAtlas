#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

from promoter_atlas.prediction.expression_predictor import ExpressionPrediction

def main():
    parser = argparse.ArgumentParser(description="Predict gene expression from DNA sequences")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to FASTA file containing DNA sequences")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save predictions")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to custom model weights (optional)")
    parser.add_argument("--format", type=str, choices=["csv", "json", "tsv"],
                      default="csv", help="Output format")
    args = parser.parse_args()
    
    # Load predictor
    print(f"Loading prediction model from {args.model_path}")
    predictor = ExpressionPrediction.load(
        model_path=args.model_path
    )
    
    # Predict expression
    print(f"Predicting expression levels for sequences in {args.input}")
    predictions = predictor.predict_fasta(args.input)
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format and save results
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    else:
        # Create DataFrame
        df = pd.DataFrame(list(predictions.items()), columns=["sequence_id", "predicted_level"])
        
        # Save as CSV or TSV
        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "tsv":
            df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Predictions saved to {output_path}")
    
    # Print summary statistics
    values = list(predictions.values())
    print("\nPrediction Statistics:")
    print(f"  Number of sequences: {len(values)}")
    print(f"  Min: {min(values):.4f}")
    print(f"  Max: {max(values):.4f}")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Median: {np.median(values):.4f}")

if __name__ == "__main__":
    main()