# Neurosymbolic Reasoning on Microbial Knowledge Graph Embeddings

This directory contains tools for performing analogy reasoning on knowledge graph embeddings to analyze relationships between microbial taxa and their physical growth preferences.

## Overview

The analogy reasoning follows the vector arithmetic pattern:
```
query_taxon - physical_preference + opposite_physical_preference = predicted_taxon
```

For example:
- `E. coli - aerobe + anaerobe = ?`
- `Bacillus subtilis - high_pH + low_pH = ?`
- `Thermotoga maritima - high_temp + low_temp = ?`

## Data Sources

The analysis uses:
- **Embeddings**: `output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-04-07_03_18_35.tsv.gz` (500-dimensional node embeddings)
- **Oxygen preferences**: `output/NCBITaxon_to_oxygen.tsv`
- **Salinity preferences**: `taxa_media/NCBITaxon_to_salinity_v3.tsv`  
- **pH preferences**: `taxa_media/taxa_pH_opt_mapping_adjusted_v2.tsv`
- **Temperature preferences**: `taxa_media/NCBITaxon_to_temp_opt_v2.tsv`

## Physical Trait Categories

### Oxygen Requirements
- **aerobe** ↔ **anaerobe**
- **facultative_anaerobe** ↔ **aerobe** 
- **microaerophile** ↔ **aerobe**

### Salinity Tolerance  
- **halophilic** ↔ **non_halophilic**
- **moderately_halophilic** ↔ **non_halophilic**

### pH Optimum
- **high** ↔ **low**
- **mid1** ↔ **mid2**

### Temperature Optimum
- **high** ↔ **low**  
- **mid1** ↔ **mid4**
- **mid2** ↔ **mid3**

## Usage

### Basic Usage
```bash
cd neurosymbolreason
python analogy_reasoning.py
```

### What the Script Does

1. **Loads Embeddings**: Reads the 500-dimensional DeepWalk embeddings for all nodes
2. **Loads Trait Data**: Loads physical growth preference data for oxygen, salinity, pH, and temperature
3. **Identifies Query Taxa**: Finds all taxa with both embeddings and known traits
4. **Performs Analogies**: For each taxon-trait combination:
   - Computes trait representative vectors (mean of all taxa with that trait)
   - Performs vector arithmetic: `query - trait + opposite_trait` 
   - Finds top 10 closest **NCBITaxon:**, **strain:**, **ph_***, and **nacl_*** nodes using cosine similarity
   - Calculates self-match score for quality assessment
5. **Analyzes Results**: Creates visualizations and statistics
6. **High-Quality Analysis**: Identifies predictions with similarity scores above or statistically close to self-match

### Output Files

- `analogy_reasoning_results.csv`: Complete results for all queries
- `high_quality_matches_detailed.csv`: Predictions above self-match threshold
- `high_quality_matches_summary.json`: Summary of high-quality matches
- `analysis_stats.json`: General summary statistics
- `analogy_reasoning_analysis.png/pdf`: Main visualization plots
- `high_quality_matches_analysis.png/pdf`: High-quality matches specific plots

## Results Structure

Each result contains:
- `query_taxon`: Original taxon ID (e.g., NCBITaxon:562)
- `trait_type`: Type of trait (oxygen, salinity, ph_opt, temp_opt)
- `trait_value`: Current trait value (e.g., aerobe, high, low)
- `opposite_trait`: The opposite trait used in analogy
- `rank`: Ranking of this prediction (1-10)
- `predicted_taxon`: The predicted taxon from analogy reasoning (NCBITaxon:*, strain:*, ph_*, or nacl_*)
- `similarity_score`: Cosine similarity score (0-1)
- `self_match_score`: Similarity between original query taxon and analogy result vector (taxon - phenotype1 + phenotype2)
- `above_self_match`: Boolean indicating if prediction exceeds self-match score

## Example Results

Query: `NCBITaxon:562` (E. coli) with `oxygen:aerobe`
Analogy: E. coli - aerobe + anaerobe = ?

Top predictions might include:
1. NCBITaxon:1234 (similarity: 0.89)
2. strain:bacdive_5678 (similarity: 0.85)
3. NCBITaxon:9012 (similarity: 0.82)

## Interpretation

**High-Quality Matches** are those with similarity scores above or statistically close to the self-match score:
- **Above self-match**: Predictions more similar to the analogy result vector than the original query taxon (indicating strong analogy)
- **Statistically close**: Within 95% of self-match score (indicating reasonable analogy)

**Self-Match Score**: Represents how similar the original query taxon is to the analogy result vector (`query - phenotype1 + phenotype2`). This serves as a baseline - predictions should ideally be more similar to the analogy result than the original query is.

**Node Type Distribution**:
- **NCBITaxon:** and **strain:**: Biological organisms with similar properties
- **ph_***: pH-related concept nodes (e.g., ph_acidic, ph_alkaline)
- **nacl_***: Salinity-related concept nodes (e.g., nacl_high, nacl_tolerance)

Lower quality predictions may indicate:
- Insufficient training data for that trait
- Complex biological relationships not captured in embeddings
- Need for more sophisticated analogy methods

## Extensions

The framework can be extended to:
- Include more physical traits (pH range, temperature range, etc.)
- Test different analogy formulations
- Validate predictions against known biology
- Compare with other embedding methods

## Dependencies

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Performance

- ~1.55M embeddings loaded
- Analysis covers taxa with known traits across 4 physical preference types
- Runtime: ~10-30 minutes depending on data size
- Memory usage: ~4-8GB for full embedding matrix

## Future Work

1. **Validation**: Compare predictions against experimental data
2. **Multi-trait analogies**: E.g., aerobe + high_temp - mesophile + anaerobe
3. **Semantic evaluation**: Assess biological meaningfulness of predictions
4. **Comparison studies**: Test against other knowledge graph embedding methods