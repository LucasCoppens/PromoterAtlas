# PromoterAtlas

PromoterAtlas is a transformer-based deep learning model for understanding and engineering bacterial regulatory sequences, with applications in bacterial biology, synthetic biology, and comparative genomics.

## Overview

PromoterAtlas is a 1.8 million parameter transformer model trained on approximately 9 million regulatory sequences from over 3,000 gammaproteobacterial species. The model demonstrates cross-species recognition of diverse regulatory elements, including:

- Ribosomal binding sites
- Various bacterial promoter types (σ70, σ38, σ54, σ32, σ28, σ24)
- Transcription factor binding sites
- Terminators

## Key Features

- **Cross-species regulatory element recognition**: Identifies conserved regulatory features across the gammaproteobacteria clade
- **Promoter annotation**: Accurate in silico identification of regulatory features across gammaproteobacteria
- **Expression prediction**: State-of-the-art performance in predicting transcription and translation levels

## Model Architecture

PromoterAtlas uses a custom DNATransformer architecture combining:
- Convolutional filters for local pattern recognition
- Rotary attention blocks for capturing long-range dependencies
- Feed-forward layers with residual connections

## System Requirements

### Software Dependencies
- Python 3.10 or 3.11
- PyTorch >= 2.0.0
- BioPython >= 1.79
- NumPy < 2.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- h5py >= 3.0.0
- logomaker
- bcbio-gff

### Operating Systems
- Tested on: macOS 14+, Ubuntu 22.04
- Should work on: Any OS supporting Python 3.10+ and PyTorch

### Hardware
- **Minimum:** 8 GB RAM, any modern CPU
- **Recommended:** 16 GB RAM, CUDA-compatible GPU for faster inference
- **Disk space:** ~200 MB for models and dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/LucasCoppens/PromoterAtlas.git
cd PromoterAtlas

# Install the package
pip install -e .
```

**Typical install time:** 3-5 minutes on a normal desktop computer.

## Model Weights

The `trained_model_weights` directory contains weights for various models trained in this work:

| Weights File | Architecture | Description |
| ------------ | ------------ | ----------- |
| promoteratlas-base.pt | DNATransformer | Base model trained on ~9M regulatory sequences |
| promoteratlas-annotation.pt | AnnotationSegmenter | Model for annotating regulatory elements |
| promoteratlas-trspred-lafleur2022.pt | TrsPredModel | Transcription prediction based on Lafleur et al. 2022 |
| promoteratlas-trspred-hossain2020.pt | TrsPredModel | Transcription prediction based on Hossain et al. 2020 |
| promoteratlas-trspred-urtecho2018.pt | TrsPredModel | Transcription prediction based on Urtecho et al. 2018 |
| promoteratlas-trspred-yu2021.pt | TrsPredModel | Transcription prediction based on Yu et al. 2021 |
| promoteratlas-tslpred-kosuri2013.pt | TrsPredModel | Translation prediction based on Kosuri et al. 2013 |

## Demo

### Quick Start

Annotate a bacterial genome with promoter predictions:

```bash
python scripts/annotate.py --input data/genomes/NC_000913.1.gb --output NC_000913.1_annotated_test.gb
```

### Expected Output

The output GenBank file will contain additional features of type "regulatory" with qualifiers indicating:
- Sigma factor type (σ70, σ54, σ32, σ28, σ24)
- -10 and -35 box positions (or -12 and -24 for σ54)
- Ribosomal binding sites (RBS)

### Demo Run Time
- **E. coli genome (~4.6 Mb):** ~2-3 minutes on CPU, ~30 seconds with GPU
- **Smaller genomes:** Proportionally faster

### Expected Demo Output

Running on a typical gammaproteobacterial genome should identify:
- ~300-400 σ70/σ38 promoters (precision-optimized default settings)
- ~15-50 alternative sigma factor promoters
- Results validated against Cho et al. 2014 ChIP-seq data

## Usage Examples

### Annotate Promoter Elements

Supports GenBank, GFF3, and FASTA formats:

```bash
# GenBank
python scripts/annotate.py --input genome.gb --output annotated.gb

# GFF3 (requires separate FASTA reference)
python scripts/annotate.py --input genome.gff3 --reference genome.fasta --output annotated.gff3

# FASTA (direct sequence annotation, outputs TSV)
python scripts/annotate.py --input sequences.fasta --output annotations.tsv
```

#### FASTA Annotation with Sliding Windows

When annotating FASTA files directly, sequences are processed differently than GenBank/GFF3 files:

- **Minimum length**: All sequences must be at least 200bp (the model's input size)
- **Sliding window**: Sequences longer than 200bp are processed using a sliding window approach
- **Output format**: Results are saved as TSV (or JSON with `--output-format json`)

The sliding window approach extracts overlapping 200bp windows from longer sequences:

```
Example: 230bp sequence with step_size=20

Window 1: positions   0-199  (200bp)
Window 2: positions  20-219  (200bp)
Window 3: positions  30-229  (200bp, last frame always included)
```

**FASTA-specific options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--step-size` | 20 | Step size (bp) between consecutive windows |
| `--merge-overlaps` | off | Merge overlapping predictions using consensus voting |

**Examples:**

```bash
# Basic FASTA annotation
python scripts/annotate.py --input sequences.fasta --output annotations.tsv

# Custom step size (smaller = more overlap, higher accuracy, slower)
python scripts/annotate.py --input sequences.fasta --output annotations.tsv --step-size 10

# Merge overlapping predictions into consensus calls
python scripts/annotate.py --input sequences.fasta --output annotations.tsv --merge-overlaps

# JSON output format
python scripts/annotate.py --input sequences.fasta --output annotations.json --output-format json
```

**Output format (TSV):**

Without `--merge-overlaps`:
```
sequence_id  window_start  window_end  element_type                   element_start  element_end  abs_start  abs_end
seq1         0             200         σ70 / σ38 promoter -10 element 145            151          145        151
```

With `--merge-overlaps`:
```
sequence_id  element_type                    start  end  length
seq1         σ70 / σ38 promoter -10 element  145    151  6
```

### Visualize DNA Sequence Features

```bash
python scripts/make_logit_plot.py --genbank your_genome.gb --locus-tag your_gene_tag --output gene_plot.png
```
To include the attention map visualisation, add `--attention-map`

### Predict Gene Expression

```bash
python scripts/predict_expression.py --input sequences.fasta --output predictions.csv --model-path trained_model_weights/promoteratlas-trspred-lafleur2022.pt
```

##### Note 1: The expression prediction models are trained on 86 nc sequences so input sequences for inference should have a length of 86 as well.
##### Note 2: Our model outputs negative values where more negative = stronger promoter. 

### Train Base Model

```bash
python scripts/train_base_model.py --n_point_masks 20 --batch_size 1024 --data_path data/processed/sequence_dataset.h5
```

## Data Availability

The data used to train the base model is available at:
https://huggingface.co/datasets/LCoppens/PromoterAtlas-data

## Citation & Paper

For details on the model architecture, training, and validation, see the accompanying paper:

**PromoterAtlas: decoding regulatory sequences across Gammaproteobacteria using a transformer model**

Code to reproduce all analyses and figures is available at [github.com/LucasCoppens/PromoterAtlas_paper](https://github.com/LucasCoppens/PromoterAtlas_paper).

## License

MIT License - see [LICENSE](LICENSE) file for details.
