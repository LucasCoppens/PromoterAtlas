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

## Installation

```bash
# Clone the repository
git clone https://github.com/LucasCoppens/PromoterAtlas.git
cd PromoterAtlas

# Install the package
pip install -e .
```

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

## Usage Examples

### Annotate Promoter Elements in a GenBank File

```bash
python scripts/annotate_genbank.py --input your_genome.gb --output annotated_genome.gb
```

### Visualize DNA Sequence Features

```bash
python scripts/make_logit_plot.py --genbank your_genome.gb --locus-tag your_gene_tag --output gene_plot.png
```

### Predict Gene Expression

```bash
python scripts/predict_expression.py --input sequences.fasta --output predictions.csv --model-path trained_model_weights/promoteratlas-trspred-lafleur2022.pt
```

Note 1: The expression prediction models are trained on 86 nc sequences so input sequences for inference should have a length of 86 as well.\n
Note 2: Our model outputs negative values where more negative = stronger promoter. 

### Train Base Model

```bash
python scripts/train_base_model.py --n_point_masks 20 --batch_size 1024 --data_path data/processed/sequence_dataset.h5
```

## Data Availability

The data used to train the base model is available at:
https://huggingface.co/datasets/LCoppens/PromoterAtlas-data
