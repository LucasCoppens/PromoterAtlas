import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import multiprocessing
from functools import partial
from Bio import SeqIO

from promoter_atlas.utils.genomics import extract_promoter_regions, sequence_to_onehot

class MotifClusterFinder:
    """Find clusters of similar motifs across promoter regions."""
    
    def __init__(self, model=None, device='cpu'):
        """Initialize with a DNA transformer model."""
        if model is None:
            from promoter_atlas.models.dna_transformer import DNATransformer
            model_path = Path("trained_weights/base_model/promoteratlas-base.pt")
            if not model_path.exists():
                model_path = Path("trained_model_weights/promoteratlas-base.pt")
            
            model = DNATransformer()
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def extract_filtered_windows(self, promoter_regions: List[Dict], 
                               window_size: int = 24, 
                               baselogit_threshold: float = 0.8) -> Dict:
        """Extract embeddings for windows with high baselogit scores."""
        promoter_embeddings = {}
        
        for promoter in promoter_regions:

            # Convert sequence to one-hot encoding
            x = sequence_to_onehot(promoter['sequence']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get both output logits and latent embeddings
                output, _, latent = self.model(x, return_latent=True)
                
            locus_tag = promoter['locus_tag']
            promoter_embeddings[locus_tag] = {
                'gene': promoter['gene'] if promoter['gene'] else 'Unknown',
                'organism': promoter.get('organism', 'Unknown'),
                'strand': promoter.get('strand', '+'),
                'windows': []
            }
            
            # Apply sliding window to find high-baselogit regions
            seq_length = len(promoter['sequence'])
            for i in range(seq_length - window_size + 1):
                window_seq = promoter['sequence'][i:i+window_size]
                
                # Calculate average baselogit for actual input bases
                input_baselogits = []
                for pos, base in enumerate(window_seq):
                    if base in {'A', 'C', 'G', 'T'}:
                        base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[base]
                        input_baselogits.append(output[0, base_idx, i+pos].item())
                
                # Skip if no valid bases or average baselogit below threshold
                if not input_baselogits or sum(input_baselogits)/len(input_baselogits) < baselogit_threshold:
                    continue
                
                # Extract embedding for this window
                window_embedding = latent[0, :, i:i+window_size]
                avg_embedding = torch.mean(window_embedding, dim=1).cpu().numpy()
                
                # Store window info
                window_info = {
                    'position': i - seq_length,  # Position relative to gene start
                    'embedding': avg_embedding,
                    'avg_baselogit': sum(input_baselogits)/len(input_baselogits)
                }
                
                promoter_embeddings[locus_tag]['windows'].append(window_info)
        
        # Filter out promoters with no high-baselogit windows
        filtered_promoters = {k: v for k, v in promoter_embeddings.items() if v['windows']}
        print(f"Found {len(filtered_promoters)} promoters with windows above baselogit threshold {baselogit_threshold}")
        
        return filtered_promoters
    
    def compute_distance_matrix(self, filtered_promoters: Dict, 
                              n_jobs: int = None) -> Tuple[np.ndarray, List[str]]:
        """Compute distance matrix between promoters based on minimum window distance."""
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 2)
            
        locus_tags = list(filtered_promoters.keys())
        n_promoters = len(locus_tags)
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_promoters, n_promoters))
        
        # Prepare task list for parallelization
        tasks = []
        for i in range(n_promoters):
            windows1 = [w['embedding'] for w in filtered_promoters[locus_tags[i]]['windows']]
            for j in range(i+1, n_promoters):
                windows2 = [w['embedding'] for w in filtered_promoters[locus_tags[j]]['windows']]
                tasks.append((i, j, windows1, windows2))
        
        # Use parallel processing to compute distances
        print(f"Computing pairwise distances using {n_jobs} parallel jobs...")
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap(self._compute_distance_for_pair, tasks), 
                               total=len(tasks), 
                               desc="Computing distances"))
        
        # Fill distance matrix with results
        for i, j, min_distance in results:
            distance_matrix[i, j] = min_distance
            distance_matrix[j, i] = min_distance  # Symmetric
        
        return distance_matrix, locus_tags
    
    @staticmethod
    def _compute_distance_for_pair(args):
        """Compute minimum distance between two promoters' windows using cosine distance."""
        i, j, windows1, windows2 = args
        
        # Find minimum distance between any pair of windows
        min_distance = float('inf')
        for window1 in windows1:
            for window2 in windows2:
                # Use cosine distance
                dot_product = np.dot(window1, window2)
                norm1 = np.linalg.norm(window1)
                norm2 = np.linalg.norm(window2)
                
                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    dist = 1.0  # Maximum distance when a vector is zero
                else:
                    similarity = dot_product / (norm1 * norm2)
                    # Ensure similarity is within [-1, 1] due to floating point errors
                    similarity = max(min(similarity, 1.0), -1.0)
                    dist = 1.0 - similarity
                
                min_distance = min(min_distance, dist)
        
        return (i, j, min_distance)
    
    def analyse_genbank(self, genbank_path: str, window_size: int = 24, 
                     baselogit_threshold: float = 0.8, n_jobs: int = None,
                     distance_file: Optional[Path] = None) -> Tuple[np.ndarray, List[str], Dict]:
        """Process a GenBank file to find motif clusters."""

        # Load genome and extract promoters
        with open(genbank_path) as f:
            gbs = list(SeqIO.parse(f, "genbank"))
        promoters = extract_promoter_regions(gbs)
        print(f"Found {len(promoters)} promoter regions in {genbank_path}")
        
        # Check for existing distance file
        if distance_file is not None and distance_file.exists():
            print(f"Loading precomputed distances from {distance_file}")
            data = np.load(distance_file)
            distances = data['distances']
            locus_tags = data['locus_tags']
            
            # Convert locus_tags to list if numpy array
            if isinstance(locus_tags, np.ndarray):
                locus_tags = locus_tags.tolist()
                
            # Create filtered promoters dictionary with metadata
            filtered_promoters = {}
            for lt in locus_tags:
                for p in promoters:
                    if p['locus_tag'] == lt:
                        filtered_promoters[lt] = {
                            'gene': p['gene'] if p['gene'] else 'Unknown',
                            'organism': p.get('organism', 'Unknown'),
                            'strand': p.get('strand', '+'),
                            'windows': []  # No need for actual windows when using distance file
                        }
                        break
        else:
            # Extract filtered windows and compute distances
            filtered_promoters = self.extract_filtered_windows(
                promoters, window_size, baselogit_threshold)
            
            if not filtered_promoters:
                raise ValueError(f"No promoters found with windows above threshold {baselogit_threshold}")
                
            distances, locus_tags = self.compute_distance_matrix(filtered_promoters, n_jobs)
            
            # Save distance matrix if not provided
            if distance_file is not None:
                distance_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    distance_file, 
                    distances=distances, 
                    locus_tags=np.array(locus_tags)
                )
                print(f"Distances saved to {distance_file}")
        
        return distances, locus_tags, filtered_promoters
    
    def cluster_promoters(self, distances: np.ndarray, min_samples: int = 5, 
                        eps: float = 0.5, n_components: int = 2,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply UMAP dimensionality reduction followed by DBSCAN clustering."""
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Apply UMAP dimensionality reduction
        print("Applying UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=n_components,
            metric='precomputed',
            random_state=random_state
        )
        umap_embedding = reducer.fit_transform(distances)
        
        # Apply DBSCAN clustering to the UMAP embedding
        print("Applying DBSCAN clustering to UMAP embedding...")
        clusterer = DBSCAN(
            eps=eps,  # Maximum distance between samples in a cluster
            min_samples=min_samples,  # Min samples in neighborhood for core point
            n_jobs=-1
        )
        cluster_labels = clusterer.fit_predict(umap_embedding)
        
        # Count clusters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return umap_embedding, cluster_labels

    def generate_cluster_report(self, locus_tags: List[str], filtered_promoters: Dict, 
                               cluster_labels: np.ndarray, output_file: Path) -> pd.DataFrame:
        """Generate a CSV report of clusters with gene information."""
        rows = []
        for i, (locus_tag, label) in enumerate(zip(locus_tags, cluster_labels)):
            if label == -1:  # Skip noise points
                continue
                
            promoter = filtered_promoters[locus_tag]
            
            # Find window with highest baselogit score if available
            best_pos = None
            best_score = 0
            
            if promoter['windows']:
                best_window = max(promoter['windows'], key=lambda w: w['avg_baselogit'])
                best_pos = best_window['position']
                best_score = best_window['avg_baselogit']
            
            rows.append({
                'cluster_id': label,
                'locus_tag': locus_tag,
                'gene': promoter['gene'],
                'organism': promoter.get('organism', 'Unknown'),
                'strand': promoter.get('strand', '+'),
                'position': best_pos,
                'avg_baselogit': best_score
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        df.to_csv(output_file, index=False)
        print(f"Cluster report written to {output_file}")
        
        return df