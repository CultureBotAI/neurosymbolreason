# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements neurosymbolic analogy reasoning on microbial knowledge graph embeddings. The core technique uses vector arithmetic to predict relationships between microbial taxa and their physical growth preferences:

```
query_taxon - physical_preference + opposite_physical_preference = predicted_taxon
```

Example: `E. coli - aerobe + anaerobe = ?` finds taxa similar to E. coli but with anaerobic preferences.

The system analyzes relationships across four physical trait types:
- **Oxygen requirements**: aerobe ↔ anaerobe, facultative_anaerobe ↔ aerobe, microaerophile ↔ aerobe
- **Salinity tolerance**: halophilic ↔ non_halophilic, moderately_halophilic ↔ non_halophilic
- **pH optimum**: high ↔ low, mid1 ↔ mid2
- **Temperature optimum**: high ↔ low, mid1 ↔ mid4, mid2 ↔ mid3

## Dependencies

**Installation**:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

**For ultra-fast implementation** (optional):
```bash
pip install faiss-cpu annoy  # or faiss-gpu for GPU acceleration
```

**Python version**: Python 3.7+ (tested with 3.10+)

## Architecture

### Three Implementations (Performance Tiers)

The codebase provides three progressively optimized implementations:

1. **`analogy_reasoning.py`** - Original baseline
   - Standard sklearn cosine_similarity
   - ~5-10 queries/second
   - Use for: development, accuracy verification, small datasets

2. **`analogy_reasoning_optimized.py`** - Production-ready
   - Pre-normalized embeddings + dot product (5-10x speedup)
   - Trait vector caching (10-50x speedup)
   - Pre-filtering target indices (2-5x speedup)
   - ~50-150 queries/second
   - Use for: production workloads, medium datasets (<1M nodes)

3. **`analogy_reasoning_ultra_fast.py`** - Large-scale
   - Approximate Nearest Neighbors (FAISS/Annoy) (10-100x speedup)
   - Optional float16 memory optimization
   - ~500-2000 queries/second
   - Use for: large datasets (>1M nodes), real-time applications
   - Requires: `pip install faiss-cpu annoy`

### Core Class: AnalogyReasoner

The `AnalogyReasoner` class handles the complete pipeline:

**Initialization**:
```python
reasoner = AnalogyReasoner(
    embeddings_path="path/to/embeddings.tsv.gz",
    data_dir="../",        # Directory with trait data files
    output_dir="./"        # Where to save results
)
```

**Key Methods**:
- `load_embeddings()`: Loads DeepWalk embeddings, creates embedding matrix
- `load_trait_data()`: Loads oxygen, salinity, pH, temperature preference mappings
- `get_taxa_with_traits()`: Returns taxa that have both embeddings and trait data
- `perform_analogy_reasoning(query_taxon, trait_type, trait_value)`: Executes analogy
  - Computes trait representative vectors (mean of all taxa with that trait)
  - Performs vector arithmetic: `query - trait + opposite_trait`
  - Finds top 10 closest nodes using cosine similarity
  - Returns predictions with similarity scores and self-match baseline
- `run_all_analogies()`: Batch processes all taxa-trait combinations

**Key Architectural Concepts**:
- **Self-match score**: Similarity between original query taxon and analogy result vector. Serves as quality baseline - predictions should exceed this.
- **Trait opposites mapping**: Dictionary defining opposite trait pairs (e.g., aerobe↔anaerobe)
- **Trait representative vectors**: Mean embedding of all taxa with a given trait value
- **Node filtering**: Results filtered to NCBITaxon:*, strain:*, ph_*, nacl_* nodes only

## Data Dependencies

**Note**: Data files are not included in this repository (excluded via .gitignore due to size). You must provide your own embedding and trait data files.

The system requires embeddings and trait data files. Paths are relative to `data_dir`:

**Embeddings** (required):
```
output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-04-07_03_18_35.tsv.gz
```
- Format: TSV.gz with header, first column = node_id, remaining columns = embedding dimensions
- ~1.55M nodes with 500-dimensional embeddings
- File size: ~20MB compressed

**Trait Data Files** (paths relative to data_dir):
- Oxygen: `output/NCBITaxon_to_oxygen.tsv`
- Salinity: `taxa_media/NCBITaxon_to_salinity_v3.tsv`
- pH: `taxa_media/taxa_pH_opt_mapping_adjusted_v2.tsv`
- Temperature: `taxa_media/NCBITaxon_to_temp_opt_v2.tsv`

When updating data paths, modify the path references in the `load_trait_data()` or `load_trait_data_from_kg()` methods.

## Running the Code

### Full Analysis Pipeline

Run complete analogy reasoning on all taxa-trait combinations:

```bash
# Original implementation
python analogy_reasoning.py

# Optimized (recommended for most use cases)
python analogy_reasoning_optimized.py

# Ultra-fast (for large-scale analysis)
# Requires: pip install faiss-cpu annoy
python analogy_reasoning_ultra_fast.py
```

**Runtime**: 10-30 minutes for original, faster for optimized versions
**Memory**: 4-8GB for full embedding matrix

**Output Files Generated**:
- `analogy_reasoning_results.csv` - All query results with similarity scores
- `high_quality_matches_detailed.csv` - Predictions above self-match threshold
- `high_quality_matches_summary.json` - Summary statistics
- `analysis_stats.json` - General statistics
- `analogy_reasoning_analysis.png/pdf` - Visualization plots
- `high_quality_matches_analysis.png/pdf` - High-quality match analysis

### Example/Interactive Usage

For testing specific queries or understanding the API:

```bash
python example_usage.py
```

This script demonstrates:
- Loading data and initializing the reasoner
- Running specific taxon-trait queries
- Interpreting prediction results and self-match scores
- Checking if specific taxa have required data

### Performance Benchmarking

Test different optimization strategies on your hardware:

```bash
python performance_benchmark.py
```

Tests: cosine similarity methods, filtering strategies, memory usage patterns, parallel processing

### Testing

```bash
python test_analogy_fix.py
```

Tests specific analogy reasoning fixes and edge cases.

## Development Notes

### Modifying Trait Opposites

To add or modify trait opposite mappings, edit the `base_trait_opposites` dictionary in `AnalogyReasoner.__init__()`. The system automatically computes quantitative trait opposites (high↔low) for pH, temperature, etc.

### Performance Optimization Strategy

When optimizing or debugging performance issues:

1. Profile with `performance_benchmark.py` to identify bottlenecks
2. For medium datasets, start with `analogy_reasoning_optimized.py`
3. Only use ANN methods (ultra_fast) when dataset size demands it
4. Key optimization techniques already implemented:
   - Pre-normalization eliminates redundant norm calculations
   - Trait vector caching avoids recomputation
   - Pre-filtering reduces similarity computation scope
   - See PERFORMANCE_OPTIMIZATION_GUIDE.md for detailed explanations

### Understanding Results

**CSV Structure**:
- `query_taxon`: Original taxon NCBITaxon:* ID
- `trait_type`: oxygen, salinity, ph_opt, or temp_opt
- `trait_value`: Current trait value (e.g., "aerobe")
- `opposite_trait`: Opposite trait used in analogy
- `rank`: 1-10 ranking of prediction
- `predicted_taxon`: Predicted node (NCBITaxon:*, strain:*, ph_*, nacl_*)
- `similarity_score`: Cosine similarity to analogy result vector
- `self_match_score`: Baseline similarity (query taxon vs analogy vector)
- `above_self_match`: Boolean indicating prediction quality

**Quality Interpretation**:
- Predictions with `similarity_score > self_match_score` are high-quality
- Self-match score represents how well the original taxon matches the analogy
- Node types indicate different prediction meanings:
  - NCBITaxon:/strain: - Biological organisms
  - ph_*/nacl_* - Concept nodes representing environmental conditions

### Adding New Physical Traits

To extend the system with additional traits:

1. Add trait data loading in `load_trait_data()` method
2. Define trait opposites in `base_trait_opposites` (categorical) or add to `quantitative_traits` list
3. Update the analogy reasoning loop in `run_all_analogies()` to include the new trait type
4. Ensure trait data files follow the same TSV format: taxon_id\ttrait_value
