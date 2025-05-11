#!/usr/bin/env python
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import numpy as np
import pandas as pd

from promoter_atlas.clustering.cluster_finder import MotifClusterFinder

def plot_clusters(umap_embedding, cluster_labels, locus_tags, filtered_promoters, 
                 title="Gene Clusters based on Promoter Motifs", output_path=None):
    """Create a visualization of the cluster analysis."""
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    plt.figure(figsize=(12, 10))
    
    # Create color map for clusters
    cmap = plt.cm.get_cmap('viridis', max(2, n_clusters))
    
    # Plot noise points
    noise_mask = cluster_labels == -1
    plt.scatter(umap_embedding[noise_mask, 0], 
                umap_embedding[noise_mask, 1], 
                c='lightgrey', 
                s=10,
                alpha=0.5,
                label='Unclustered')
    
    # Plot clustered points with cluster-based colors
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise points
            continue
            
        # Create mask for this cluster
        cluster_mask = cluster_labels == cluster_id
        
        # Plot points for this cluster
        plt.scatter(umap_embedding[cluster_mask, 0], 
                   umap_embedding[cluster_mask, 1], 
                   c=[cmap(cluster_id % cmap.N)],  # Use modulo to handle more clusters than colors
                   s=30,
                   label=f'Cluster {cluster_id}',
                   alpha=0.8)
    
    plt.legend(title="Clusters", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Find gene clusters based on promoter motifs")
    parser.add_argument("--genbank", type=str, required=True,
                       help="Path to GenBank file")
    parser.add_argument("--window-size", type=int, default=24,
                       help="Size of sliding window for motif analysis (default: 24)")
    parser.add_argument("--baselogit-threshold", type=float, default=0.8,
                       help="Minimum average baselogit score for windows (default: 0.8)")
    parser.add_argument("--distance-file", type=str,
                       help="Path to precomputed distance matrix (optional)")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum samples parameter for DBSCAN (default: 5)")
    parser.add_argument("--eps", type=float, default=0.5,
                       help="Maximum distance between samples in a cluster (default: 0.5)")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs for distance computation")
    parser.add_argument("--output-prefix", type=str, default=None,
                       help="Prefix for output files (default: genbank accession)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None)")
    args = parser.parse_args()
    
    # Determine accession
    accession = Path(args.genbank).stem
    
    # Setup output paths
    output_prefix = args.output_prefix or accession
    
    # Setup distance file path
    distance_file = None
    if args.distance_file:
        distance_file = Path(args.distance_file)
    else:
        # Create default path in data directory
        data_dir = Path("data/distances")
        data_dir.mkdir(exist_ok=True, parents=True)
        distance_file = data_dir / f"{accession}_w{args.window_size}_t{args.baselogit_threshold:.2f}_distances.npz"
    
    # Initialize finder
    finder = MotifClusterFinder()
    
    # Analyze GenBank file
    distances, locus_tags, filtered_promoters = finder.analyse_genbank(
        genbank_path=args.genbank,
        window_size=args.window_size,
        baselogit_threshold=args.baselogit_threshold,
        n_jobs=args.n_jobs,
        distance_file=distance_file
    )
    
    # Cluster promoters
    umap_embedding, cluster_labels = finder.cluster_promoters(
        distances=distances,
        min_samples=args.min_samples,
        eps=args.eps,
        random_state=args.seed
    )
    
    # Generate cluster report
    output_file = Path(f"{output_prefix}_clusters.csv")
    cluster_df = finder.generate_cluster_report(
        locus_tags=locus_tags,
        filtered_promoters=filtered_promoters,
        cluster_labels=cluster_labels,
        output_file=output_file
    )
    
    # Plot results
    plot_clusters(
        umap_embedding=umap_embedding,
        cluster_labels=cluster_labels,
        locus_tags=locus_tags,
        filtered_promoters=filtered_promoters,
        title=f"Gene Clusters in {accession} based on Promoter Motifs",
        output_path=f"{output_prefix}_clusters.png"
    )
    
    # Save UMAP embeddings with cluster labels
    umap_df = pd.DataFrame({
        'locus_tag': locus_tags,
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'cluster': cluster_labels
    })
    umap_file = Path(f"{output_prefix}_umap.csv")
    umap_df.to_csv(umap_file, index=False)
    print(f"UMAP embeddings saved to {umap_file}")
    
    # Summary of results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print("\nSummary:")
    print(f"- Analyzed {len(locus_tags)} promoters from {accession}")
    print(f"- Found {n_clusters} gene clusters")
    print(f"- Results saved with prefix '{output_prefix}'")
    print(f"- Full cluster report: {output_file}")

if __name__ == "__main__":
    main()