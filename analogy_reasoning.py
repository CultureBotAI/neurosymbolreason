#!/usr/bin/env python3
"""
Analogy Reasoning on Node Embeddings for Microbial Physical Growth Preferences

This script performs analogy reasoning on knowledge graph embeddings to test relationships
between microbial taxa and their physical growth preferences (oxygen, salinity, pH, temperature).

The analogy reasoning follows the pattern:
query_taxon - physical_pref + opposite_physical_pref = predicted_taxon

For example:
E. coli - aerobe + anaerobe = ?

Author: Generated with Claude Code
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import gzip
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from scipy import stats
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalogyReasoner:
    """
    Performs analogy reasoning on knowledge graph embeddings for microbial growth preferences.
    """
    
    def __init__(self, embeddings_path: str, data_dir: str = "../", output_dir: str = "./"):
        """
        Initialize the AnalogyReasoner.
        
        Args:
            embeddings_path: Path to the DeepWalk embeddings file (.tsv.gz)
            data_dir: Directory containing trait data files  
            output_dir: Directory to save results
        """
        self.embeddings_path = embeddings_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data containers
        self.embeddings = {}
        self.embedding_matrix = None
        self.node_list = []
        
        # Trait data
        self.oxygen_data = {}
        self.salinity_data = {}  
        self.ph_data = {}
        self.temp_data = {}
        self.ph_range_data = {}
        self.ph_delta_data = {}
        self.nacl_range_data = {}
        self.nacl_delta_data = {}
        self.nacl_opt_data = {}
        self.motility_data = {}
        self.temp_range_data = {}
        self.temp_delta_data = {}
        self.cell_width_data = {}
        self.cell_length_data = {}
        
        # Base trait opposites mapping (will be updated after data loading)
        self.base_trait_opposites = {
            'oxygen': {
                'aerobe': 'anaerobe',
                'anaerobe': 'aerobe',
                'obligate_aerobe': 'obligate_anaerobe',
                'obligate_anaerobe': 'obligate_aerobe', 
                'facultative_aerobe': 'facultative_anaerobe',
                'facultative_anaerobe': 'facultative_aerobe',
                'microaerophile': 'anaerobe'  # kept existing mapping
            },
            'salinity': {
                'halophilic': 'non_halophilic',
                'non_halophilic': 'halophilic',
                'moderately_halophilic': 'non_halophilic'
            },
            'motility': {
                'motile': 'non_motile',
                'non_motile': 'motile'
            }
        }
        
        # Quantitative traits that need dynamic middle term exclusion
        self.quantitative_traits = [
            'ph_opt', 'temp_opt', 'ph_range', 'ph_delta', 'nacl_range', 
            'nacl_delta', 'nacl_opt', 'temp_range', 'temp_delta', 
            'cell_width', 'cell_length'
        ]
        
        # Final trait opposites (will be populated after data loading)
        self.trait_opposites = {}
        
    def load_embeddings(self):
        """Load DeepWalk embeddings from compressed TSV file."""
        logger.info(f"Loading embeddings from {self.embeddings_path}")
        
        embeddings_dict = {}
        node_list = []
        
        with gzip.open(self.embeddings_path, 'rt') as f:
            # Skip header
            header = f.readline().strip().split('\t')
            embedding_dim = len(header) - 1
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    logger.info(f"Loaded {line_num} embeddings...")
                    
                parts = line.strip().split('\t')
                node_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                embeddings_dict[node_id] = embedding
                node_list.append(node_id)
        
        logger.info(f"Loaded {len(embeddings_dict)} embeddings")
        
        # Create embedding matrix for efficient similarity computation
        self.embedding_matrix = np.array([embeddings_dict[node] for node in node_list])
        self.embeddings = embeddings_dict
        self.node_list = node_list
        
    def load_trait_data_from_kg(self):
        """Load physical growth preference data directly from knowledge graph."""
        logger.info("Loading trait data directly from knowledge graph...")
        
        # Define KG file paths (adjust paths as needed for local vs HPC)
        try:
            from glob import glob
            
            # Try multiple possible paths
            possible_paths = [
                # HPC paths
                "/global/cfs/cdirs/m4689/master/kg-microbe/data/merged/*/merged-kg_nodes.tsv",
                "/global/cfs/cdirs/m4689/master/kg-microbe/data/merged/20250222/merged-kg_nodes.tsv",
                # Absolute path based on your system
                "/Users/marcin/Documents/VIMSS/ontology/KG-Hub/KG-Microbe/kg-microbe/data/merged/20250222/merged-kg_nodes.tsv",
                "/Users/marcin/Documents/VIMSS/ontology/KG-Hub/KG-Microbe/kg-microbe/data/merged/*/merged-kg_nodes.tsv",
                # Local paths relative to current directory
                str(self.data_dir / "../data/merged/*/merged-kg_nodes.tsv"),  
                str(self.data_dir / "../data/merged/20250222/merged-kg_nodes.tsv"),
                # Additional relative paths
                str(self.data_dir / "../../kg-microbe/data/merged/*/merged-kg_nodes.tsv"),
                str(self.data_dir / "../../kg-microbe/data/merged/20250222/merged-kg_nodes.tsv"),
                # Direct relative paths
                "../../kg-microbe/data/merged/20250222/merged-kg_nodes.tsv",
                "../kg-microbe/data/merged/20250222/merged-kg_nodes.tsv",
                # Path going up from neurosymbolreason directory
                "../../../kg-microbe/data/merged/20250222/merged-kg_nodes.tsv",
                "../../../../kg-microbe/data/merged/20250222/merged-kg_nodes.tsv"
            ]
            
            node_files = []
            edge_files = []
            
            for path_pattern in possible_paths:
                nodes = glob(path_pattern)
                edges = glob(path_pattern.replace('nodes', 'edges'))
                if nodes and edges:
                    node_files.extend(nodes)
                    edge_files.extend(edges)
            
            if not node_files or not edge_files:
                logger.error("Could not find merged KG files. Falling back to individual files.")
                logger.info("Searched paths:")
                for path in possible_paths:
                    logger.info(f"  - {path}")
                self.load_trait_data()  # Fallback to original method
                return
                
            # Use the most recent files
            node_path = sorted(node_files)[-1]
            edge_path = sorted(edge_files)[-1]
            
            logger.info(f"Loading KG data from: {node_path}")
            logger.info(f"Loading KG edges from: {edge_path}")
            
        except Exception as e:
            logger.error(f"Error finding KG files: {e}. Falling back to individual files.")
            self.load_trait_data()  # Fallback to original method
            return
        
        # Load nodes and edges
        try:
            nodes_df = pd.read_csv(node_path, sep='\t')
            edges_df = pd.read_csv(edge_path, sep='\t')
            logger.info(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges from KG")
            
        except Exception as e:
            logger.error(f"Error loading KG files: {e}. Falling back to individual files.")
            self.load_trait_data()  # Fallback to original method
            return
        
        # Extract trait data from KG
        self._extract_traits_from_kg(nodes_df, edges_df)
    
    def _extract_traits_from_kg(self, nodes_df, edges_df):
        """Extract trait data from knowledge graph nodes and edges."""
        # Define trait prefixes to look for
        trait_prefixes = {
            'oxygen': 'oxygen:',
            'salinity': 'salinity:', 
            'ph_opt': 'pH_opt:',
            'temp_opt': 'temp_opt:',
            'ph_range': 'pH_range:',
            'ph_delta': 'pH_delta:',
            'nacl_range': 'NaCl_range:',
            'nacl_delta': 'NaCl_delta:',
            'nacl_opt': 'NaCl_opt:',
            'motility': 'motility:',
            'temp_range': 'temp_range:',
            'temp_delta': 'temp_delta:',
            'cell_width': 'cell_width:',
            'cell_length': 'cell_length:'
        }
        
        # Initialize trait data dictionaries
        trait_data_maps = {
            'oxygen': self.oxygen_data,
            'salinity': self.salinity_data,
            'ph_opt': self.ph_data,
            'temp_opt': self.temp_data,
            'ph_range': self.ph_range_data,
            'ph_delta': self.ph_delta_data,
            'nacl_range': self.nacl_range_data,
            'nacl_delta': self.nacl_delta_data,
            'nacl_opt': self.nacl_opt_data,
            'motility': self.motility_data,
            'temp_range': self.temp_range_data,
            'temp_delta': self.temp_delta_data,
            'cell_width': self.cell_width_data,
            'cell_length': self.cell_length_data
        }
        
        # Extract trait relationships from edges
        for trait_name, prefix in trait_prefixes.items():
            # Find nodes with this trait prefix
            trait_nodes = nodes_df[nodes_df['id'].str.startswith(prefix, na=False)]
            
            if len(trait_nodes) > 0:
                # Find edges connecting NCBITaxon nodes to these trait nodes
                trait_node_ids = set(trait_nodes['id'])
                
                # Find edges where object is a trait node and subject is NCBITaxon
                trait_edges = edges_df[
                    (edges_df['object'].isin(trait_node_ids)) &
                    (edges_df['subject'].str.startswith('NCBITaxon:', na=False))
                ]
                
                # Create mapping dictionary
                trait_mapping = {}
                for _, edge in trait_edges.iterrows():
                    taxon = edge['subject']
                    trait_value = edge['object'].replace(prefix, '')  # Remove prefix
                    trait_mapping[taxon] = trait_value
                
                # Store in appropriate data structure
                trait_data_maps[trait_name].update(trait_mapping)
                
                logger.info(f"Loaded {len(trait_mapping)} {trait_name} preferences from KG")
            else:
                logger.warning(f"No {trait_name} nodes found with prefix {prefix}")
        
        # Build dynamic trait opposites after all data is loaded
        self._build_dynamic_trait_opposites()
    
    def load_trait_data(self):
        """Load physical growth preference data (fallback method)."""
        logger.info("Loading trait data from individual files...")
        
        # Load oxygen data
        oxygen_file = self.data_dir / "output" / "NCBITaxon_to_oxygen.tsv"
        if oxygen_file.exists():
            df = pd.read_csv(oxygen_file, sep='\t')
            self.oxygen_data = dict(zip(df['subject'], df['object'].str.replace('oxygen:', '')))
            logger.info(f"Loaded {len(self.oxygen_data)} oxygen preferences")
        
        # Load salinity data  
        salinity_file = self.data_dir / "taxa_media" / "NCBITaxon_to_salinity_v3.tsv"
        if salinity_file.exists():
            df = pd.read_csv(salinity_file, sep='\t')
            # Filter for NCBITaxon entries only
            df_filtered = df[df['subject'].str.startswith('NCBITaxon:')]
            self.salinity_data = dict(zip(df_filtered['subject'], df_filtered['object'].str.replace('salinity:', '')))
            logger.info(f"Loaded {len(self.salinity_data)} salinity preferences")
        
        # Load pH data
        ph_file = self.data_dir / "taxa_media" / "taxa_pH_opt_mapping_adjusted_v2.tsv"  
        if ph_file.exists():
            df = pd.read_csv(ph_file, sep='\t')
            self.ph_data = dict(zip(df['NCBITaxon'], df['pH_opt'].str.replace('pH_opt:', '')))
            logger.info(f"Loaded {len(self.ph_data)} pH preferences")
        
        # Load temperature data
        temp_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_opt_v2.tsv"
        if temp_file.exists():
            df = pd.read_csv(temp_file, sep='\t')
            self.temp_data = dict(zip(df['subject'], df['object'].str.replace('temp_opt:', '')))
            logger.info(f"Loaded {len(self.temp_data)} temperature preferences")
        
        # Load new trait data files (these may need to be created or paths adjusted)
        # pH range data
        ph_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_pH_range.tsv"
        if ph_range_file.exists():
            df = pd.read_csv(ph_range_file, sep='\t')
            self.ph_range_data = dict(zip(df['subject'], df['object'].str.replace('pH_range:', '')))
            logger.info(f"Loaded {len(self.ph_range_data)} pH range preferences")
        
        # pH delta data
        ph_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_pH_delta.tsv"
        if ph_delta_file.exists():
            df = pd.read_csv(ph_delta_file, sep='\t')
            self.ph_delta_data = dict(zip(df['subject'], df['object'].str.replace('pH_delta:', '')))
            logger.info(f"Loaded {len(self.ph_delta_data)} pH delta preferences")
        
        # NaCl range data
        nacl_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_range.tsv"
        if nacl_range_file.exists():
            df = pd.read_csv(nacl_range_file, sep='\t')
            self.nacl_range_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_range:', '')))
            logger.info(f"Loaded {len(self.nacl_range_data)} NaCl range preferences")
        
        # NaCl delta data
        nacl_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_delta.tsv"
        if nacl_delta_file.exists():
            df = pd.read_csv(nacl_delta_file, sep='\t')
            self.nacl_delta_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_delta:', '')))
            logger.info(f"Loaded {len(self.nacl_delta_data)} NaCl delta preferences")
        
        # NaCl optimal data
        nacl_opt_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_opt.tsv"
        if nacl_opt_file.exists():
            df = pd.read_csv(nacl_opt_file, sep='\t')
            self.nacl_opt_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_opt:', '')))
            logger.info(f"Loaded {len(self.nacl_opt_data)} NaCl optimal preferences")
        
        # Motility data
        motility_file = self.data_dir / "taxa_media" / "NCBITaxon_to_motility.tsv"
        if motility_file.exists():
            df = pd.read_csv(motility_file, sep='\t')
            self.motility_data = dict(zip(df['subject'], df['object'].str.replace('motility:', '')))
            logger.info(f"Loaded {len(self.motility_data)} motility preferences")
        
        # Temperature range data
        temp_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_range.tsv"
        if temp_range_file.exists():
            df = pd.read_csv(temp_range_file, sep='\t')
            self.temp_range_data = dict(zip(df['subject'], df['object'].str.replace('temp_range:', '')))
            logger.info(f"Loaded {len(self.temp_range_data)} temperature range preferences")
        
        # Temperature delta data
        temp_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_delta.tsv"
        if temp_delta_file.exists():
            df = pd.read_csv(temp_delta_file, sep='\t')
            self.temp_delta_data = dict(zip(df['subject'], df['object'].str.replace('temp_delta:', '')))
            logger.info(f"Loaded {len(self.temp_delta_data)} temperature delta preferences")
        
        # Cell width data
        cell_width_file = self.data_dir / "taxa_media" / "NCBITaxon_to_cell_width.tsv"
        if cell_width_file.exists():
            df = pd.read_csv(cell_width_file, sep='\t')
            self.cell_width_data = dict(zip(df['subject'], df['object'].str.replace('cell_width:', '')))
            logger.info(f"Loaded {len(self.cell_width_data)} cell width preferences")
        
        # Cell length data
        cell_length_file = self.data_dir / "taxa_media" / "NCBITaxon_to_cell_length.tsv"
        if cell_length_file.exists():
            df = pd.read_csv(cell_length_file, sep='\t')
            self.cell_length_data = dict(zip(df['subject'], df['object'].str.replace('cell_length:', '')))
            logger.info(f"Loaded {len(self.cell_length_data)} cell length preferences")
        
        # Build dynamic trait opposites after all data is loaded
        self._build_dynamic_trait_opposites()
    
    def _build_dynamic_trait_opposites(self):
        """Build trait opposites mapping with dynamic middle term exclusion."""
        logger.info("Building dynamic trait opposites with middle term exclusion...")
        
        # Start with base trait opposites
        self.trait_opposites = self.base_trait_opposites.copy()
        
        # Get trait data mapping
        trait_data_map = {
            'oxygen': self.oxygen_data,
            'salinity': self.salinity_data,
            'ph_opt': self.ph_data,
            'temp_opt': self.temp_data,
            'ph_range': self.ph_range_data,
            'ph_delta': self.ph_delta_data,
            'nacl_range': self.nacl_range_data,
            'nacl_delta': self.nacl_delta_data,
            'nacl_opt': self.nacl_opt_data,
            'motility': self.motility_data,
            'temp_range': self.temp_range_data,
            'temp_delta': self.temp_delta_data,
            'cell_width': self.cell_width_data,
            'cell_length': self.cell_length_data
        }
        
        # Process quantitative traits
        for trait_type in self.quantitative_traits:
            if trait_type in trait_data_map and trait_data_map[trait_type]:
                trait_values = list(set(trait_data_map[trait_type].values()))
                
                # Sort trait values to identify extremes and middle terms
                sorted_values = sorted(trait_values)
                n_values = len(sorted_values)
                
                if n_values >= 3:  # Need at least 3 values to have meaningful extremes
                    # Determine middle terms
                    middle_terms = self._get_middle_terms(sorted_values)
                    
                    # Create opposites mapping excluding middle terms
                    opposites = {}
                    
                    # Map extremes: first to last, second to second-last, etc.
                    for i in range(n_values):
                        value = sorted_values[i]
                        if value not in middle_terms:
                            opposite_idx = n_values - 1 - i
                            opposite_value = sorted_values[opposite_idx]
                            if opposite_value not in middle_terms and value != opposite_value:
                                opposites[value] = opposite_value
                    
                    self.trait_opposites[trait_type] = opposites
                    
                    logger.info(f"Trait {trait_type}: {n_values} values, middle terms excluded: {middle_terms}")
                    logger.info(f"  Opposites: {opposites}")
                
                elif n_values == 2:
                    # For binary traits, map directly
                    self.trait_opposites[trait_type] = {
                        sorted_values[0]: sorted_values[1],
                        sorted_values[1]: sorted_values[0]
                    }
                    logger.info(f"Trait {trait_type}: Binary mapping {sorted_values[0]} <-> {sorted_values[1]}")
    
    def _get_middle_terms(self, sorted_values: List[str]) -> List[str]:
        """
        Determine middle terms from sorted values.
        - Odd number of values: 1 middle term
        - Even number of values: 2 middle terms
        """
        n = len(sorted_values)
        middle_terms = []
        
        if n % 2 == 1:
            # Odd number: 1 middle term
            middle_idx = n // 2
            middle_terms.append(sorted_values[middle_idx])
        else:
            # Even number: 2 middle terms
            middle_idx1 = n // 2 - 1
            middle_idx2 = n // 2
            middle_terms.extend([sorted_values[middle_idx1], sorted_values[middle_idx2]])
        
        return middle_terms
    
    def get_taxa_with_traits(self) -> Dict[str, Dict[str, str]]:
        """
        Get all taxa that have at least one physical growth preference known.
        
        Returns:
            Dictionary mapping taxon -> {trait_type -> trait_value}
        """
        taxa_traits = defaultdict(dict)
        
        # Collect all trait data
        trait_data = {
            'oxygen': self.oxygen_data,
            'salinity': self.salinity_data, 
            'ph_opt': self.ph_data,
            'temp_opt': self.temp_data,
            'ph_range': self.ph_range_data,
            'ph_delta': self.ph_delta_data,
            'nacl_range': self.nacl_range_data,
            'nacl_delta': self.nacl_delta_data,
            'nacl_opt': self.nacl_opt_data,
            'motility': self.motility_data,
            'temp_range': self.temp_range_data,
            'temp_delta': self.temp_delta_data,
            'cell_width': self.cell_width_data,
            'cell_length': self.cell_length_data
        }
        
        for trait_type, trait_dict in trait_data.items():
            for taxon, trait_value in trait_dict.items():
                if taxon in self.embeddings:  # Only include taxa with embeddings
                    taxa_traits[taxon][trait_type] = trait_value
        
        # Filter to only include taxa with at least one trait
        taxa_traits = {taxon: traits for taxon, traits in taxa_traits.items() if traits}
        
        logger.info(f"Found {len(taxa_traits)} taxa with known traits and embeddings")
        return dict(taxa_traits)
    
    def find_closest_nodes(self, target_vector: np.ndarray, top_k: int = 10, 
                          filter_prefixes: List[str] = None, exclude_query: str = None) -> List[Tuple[str, float]]:
        """
        Find the closest nodes to a target vector using cosine similarity.
        
        Args:
            target_vector: The target embedding vector
            top_k: Number of top matches to return
            filter_prefixes: Only return nodes with these prefixes (e.g., ['NCBITaxon:', 'strain:', 'ph_', 'nacl_'])
            exclude_query: Node ID to exclude from results (usually the query taxon itself)
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        # Compute cosine similarities
        similarities = cosine_similarity([target_vector], self.embedding_matrix)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k*10]  # Get more than needed for filtering
        
        results = []
        for idx in top_indices:
            node_id = self.node_list[idx]
            similarity = similarities[idx]
            
            # Exclude query taxon if specified
            if exclude_query and node_id == exclude_query:
                continue
            
            # Apply prefix filter if specified
            if filter_prefixes:
                # Case insensitive matching for ph_ and nacl_ prefixes
                matches_prefix = False
                for prefix in filter_prefixes:
                    if prefix.lower() in ['ph_', 'nacl_']:
                        if node_id.lower().startswith(prefix.lower()):
                            matches_prefix = True
                            break
                    else:
                        if node_id.startswith(prefix):
                            matches_prefix = True
                            break
                
                if not matches_prefix:
                    continue
            
            results.append((node_id, similarity))
            
            if len(results) >= top_k:
                break
                
        return results
    
    def get_analogy_self_match_score(self, query_taxon: str, predicted_vector: np.ndarray) -> float:
        """
        Get the similarity score between the query taxon and the analogy result vector.
        This represents how well the analogy reasoning preserves the original taxon's characteristics.
        
        Args:
            query_taxon: The query taxon ID
            predicted_vector: The result of analogy reasoning (query - phenotype1 + phenotype2)
            
        Returns:
            Similarity between original query taxon and analogy result vector
        """
        if query_taxon not in self.embeddings:
            return 0.0
        
        query_vector = self.embeddings[query_taxon]
        self_similarity = cosine_similarity([query_vector], [predicted_vector])[0][0]
        return float(self_similarity)
    
    def get_raw_self_similarity(self, query_taxon: str) -> float:
        """
        Get the raw self-similarity score (taxon embedding compared to itself).
        This should always be 1.0 and serves as a reference point for visualization.
        
        Args:
            query_taxon: The query taxon ID
            
        Returns:
            Raw self-similarity (should be 1.0)
        """
        if query_taxon not in self.embeddings:
            return 0.0
        
        query_vector = self.embeddings[query_taxon]
        raw_self_similarity = cosine_similarity([query_vector], [query_vector])[0][0]
        return float(raw_self_similarity)
    
    def perform_analogy_reasoning(self, query_taxon: str, trait_type: str, 
                                 trait_value: str) -> Optional[Dict]:
        """
        Perform analogy reasoning: query_taxon - trait_value + opposite_trait_value.
        
        The analogy equation is: predicted_vector = query_vector - trait_vector + opposite_trait_vector
        All similarity comparisons (including self_match_score) are against this predicted_vector.
        
        Args:
            query_taxon: The query taxon ID
            trait_type: Type of trait (oxygen, salinity, ph_opt, temp_opt)
            trait_value: Current trait value
            
        Returns:
            Dictionary with:
            - 'predictions': List of (node_id, similarity_to_predicted_vector) tuples
            - 'self_match_score': Similarity between original query_taxon and predicted_vector
        """
        # Check if we have the necessary data
        if query_taxon not in self.embeddings:
            logger.warning(f"No embedding found for {query_taxon}")
            return None
            
        if trait_type not in self.trait_opposites:
            logger.warning(f"No opposites defined for trait type {trait_type}")
            return None
            
        if trait_value not in self.trait_opposites[trait_type]:
            logger.warning(f"No opposite defined for {trait_type}:{trait_value}")
            return None
        
        opposite_trait = self.trait_opposites[trait_type][trait_value]
        
        # Find representative vectors for trait values
        # We'll use the mean of all taxa with each trait value
        trait_vectors = defaultdict(list)
        
        trait_data_map = {
            'oxygen': self.oxygen_data,
            'salinity': self.salinity_data,
            'ph_opt': self.ph_data, 
            'temp_opt': self.temp_data,
            'ph_range': self.ph_range_data,
            'ph_delta': self.ph_delta_data,
            'nacl_range': self.nacl_range_data,
            'nacl_delta': self.nacl_delta_data,
            'nacl_opt': self.nacl_opt_data,
            'motility': self.motility_data,
            'temp_range': self.temp_range_data,
            'temp_delta': self.temp_delta_data,
            'cell_width': self.cell_width_data,
            'cell_length': self.cell_length_data
        }
        
        trait_data = trait_data_map[trait_type]
        
        for taxon, taxon_trait in trait_data.items():
            if taxon in self.embeddings:
                trait_vectors[taxon_trait].append(self.embeddings[taxon])
        
        # Compute mean vectors
        if len(trait_vectors[trait_value]) == 0 or len(trait_vectors[opposite_trait]) == 0:
            logger.warning(f"Not enough examples for {trait_type}:{trait_value} <-> {opposite_trait}")
            return None
            
        trait_vector = np.mean(trait_vectors[trait_value], axis=0)
        opposite_vector = np.mean(trait_vectors[opposite_trait], axis=0)
        
        # Perform analogy equation: predicted_vector = query_vector - trait_vector + opposite_trait_vector
        # This represents "what would the query taxon look like if it had the opposite phenotype?"
        query_vector = self.embeddings[query_taxon]
        predicted_vector = query_vector - trait_vector + opposite_vector
        
        # Find closest matches to the predicted_vector, filtering for taxa, strains, ph_, and nacl_ nodes
        # All similarity scores are computed against the predicted_vector (not the original query_vector)
        results = self.find_closest_nodes(
            predicted_vector, 
            top_k=10,
            filter_prefixes=['NCBITaxon:', 'strain:', 'ph_', 'nacl_'],
            exclude_query=query_taxon
        )
        
        # Get self-match score: similarity between original query taxon and the predicted_vector
        # This serves as a baseline - good predictions should exceed this similarity
        self_match_score = self.get_analogy_self_match_score(query_taxon, predicted_vector)
        
        # Get raw self-similarity for visualization reference
        raw_self_similarity = self.get_raw_self_similarity(query_taxon)
        
        return {
            'predictions': results,
            'self_match_score': self_match_score,
            'raw_self_similarity': raw_self_similarity
        }
    
    def run_comprehensive_analysis(self):
        """Run analogy reasoning for all taxa and traits."""
        logger.info("Starting comprehensive analogy reasoning analysis...")
        
        # Get all taxa with traits
        taxa_traits = self.get_taxa_with_traits()
        
        results = []
        
        for taxon, traits in taxa_traits.items():
            for trait_type, trait_value in traits.items():
                logger.info(f"Analyzing {taxon} with {trait_type}:{trait_value}")
                
                result_data = self.perform_analogy_reasoning(taxon, trait_type, trait_value)
                
                if result_data:
                    predictions = result_data['predictions']
                    self_match_score = result_data['self_match_score']
                    
                    for rank, (predicted_taxon, similarity) in enumerate(predictions, 1):
                        result = {
                            'query_taxon': taxon,
                            'trait_type': trait_type,
                            'trait_value': trait_value,
                            'opposite_trait': self.trait_opposites[trait_type].get(trait_value, 'unknown'),
                            'rank': rank,
                            'predicted_taxon': predicted_taxon,
                            'similarity_score': similarity,
                            'self_match_score': self_match_score,
                            'above_self_match': similarity > self_match_score
                        }
                        results.append(result)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        output_file = self.output_dir / f"analogy_reasoning_results_{self.timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(results)} results to {output_file}")
        
        # Create and save high-quality matches analysis
        self._analyze_high_quality_matches(results_df)
        
        return results_df
    
    def _analyze_high_quality_matches(self, results_df: pd.DataFrame):
        """
        Analyze and save results that have similarity scores higher than or statistically 
        close to the self-match score.
        """
        logger.info("Analyzing high-quality matches...")
        
        # Filter for matches above self-match threshold
        high_quality_df = results_df[results_df['above_self_match'] == True].copy()
        
        # Add statistical significance analysis
        # For each query, determine if predictions are statistically close to self-match
        enhanced_results = []
        
        for (query_taxon, trait_type, trait_value), group in results_df.groupby(['query_taxon', 'trait_type', 'trait_value']):
            self_match_score = group['self_match_score'].iloc[0]
            similarities = group['similarity_score'].values
            
            # Calculate statistical thresholds
            # Use 95% of self-match as "statistically close" threshold
            close_threshold = self_match_score * 0.95
            
            # Standard deviation of similarities for this query
            std_dev = np.std(similarities)
            
            for _, row in group.iterrows():
                similarity = row['similarity_score']
                
                # Determine quality categories
                above_self = similarity > self_match_score
                statistically_close = similarity > close_threshold
                in_top_percentile = row['rank'] <= 3  # Top 3 predictions
                
                enhanced_row = row.copy()
                enhanced_row['statistically_close_to_self'] = statistically_close
                enhanced_row['in_top_percentile'] = in_top_percentile
                enhanced_row['similarity_std_dev'] = std_dev
                enhanced_row['close_threshold'] = close_threshold
                
                # Only include high-quality matches
                if above_self or statistically_close:
                    enhanced_results.append(enhanced_row)
        
        if enhanced_results:
            high_quality_detailed_df = pd.DataFrame(enhanced_results)
            
            # Save detailed high-quality results
            high_quality_file = self.output_dir / f"high_quality_matches_detailed_{self.timestamp}.csv"
            high_quality_detailed_df.to_csv(high_quality_file, index=False)
            logger.info(f"Saved {len(high_quality_detailed_df)} high-quality matches to {high_quality_file}")
            
            # Create summary statistics
            summary_stats = {
                'total_queries': len(results_df['query_taxon'].unique()),
                'queries_with_high_quality_matches': len(high_quality_detailed_df['query_taxon'].unique()),
                'total_high_quality_matches': len(high_quality_detailed_df),
                'matches_above_self_match': len(high_quality_detailed_df[high_quality_detailed_df['above_self_match']]),
                'matches_statistically_close': len(high_quality_detailed_df[high_quality_detailed_df['statistically_close_to_self']]),
                'avg_self_match_score': results_df['self_match_score'].mean(),
                'high_quality_by_trait': high_quality_detailed_df.groupby('trait_type').size().to_dict(),
                'high_quality_by_node_type': {
                    'NCBITaxon': len(high_quality_detailed_df[high_quality_detailed_df['predicted_taxon'].str.startswith('NCBITaxon:')]),
                    'strain': len(high_quality_detailed_df[high_quality_detailed_df['predicted_taxon'].str.startswith('strain:')]),
                    'ph_nodes': len(high_quality_detailed_df[high_quality_detailed_df['predicted_taxon'].str.lower().str.startswith('ph_')]),
                    'nacl_nodes': len(high_quality_detailed_df[high_quality_detailed_df['predicted_taxon'].str.lower().str.startswith('nacl_')])
                }
            }
            
            # Save summary
            summary_file = self.output_dir / f"high_quality_matches_summary_{self.timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"High-quality match analysis complete. {summary_stats['queries_with_high_quality_matches']}/{summary_stats['total_queries']} queries had high-quality matches.")
        else:
            logger.warning("No high-quality matches found!")
    

    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and visualize the analogy reasoning results."""
        logger.info("Analyzing results...")
        
        # Calculate statistics
        stats = {
            'total_queries': len(results_df['query_taxon'].unique()),
            'total_predictions': len(results_df),
            'traits_analyzed': results_df['trait_type'].unique().tolist(),
            'avg_similarity_score': results_df['similarity_score'].mean(),
            'top1_avg_similarity': results_df[results_df['rank'] == 1]['similarity_score'].mean()
        }
        
        # Save statistics
        with open(self.output_dir / f"analysis_stats_{self.timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Create visualizations
        self._create_visualizations(results_df)
        
        logger.info(f"Analysis complete. Analyzed {stats['total_queries']} queries with {stats['total_predictions']} predictions.")
        
    def _create_visualizations(self, results_df: pd.DataFrame):
        """Create visualization plots."""
        plt.style.use('default')
        
        # Create main analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Top-1 similarities by trait type
        top1_df = results_df[results_df['rank'] == 1]
        trait_types = results_df['trait_type'].unique()
        
        # Create boxplot with outliers shown
        box_data = [top1_df[top1_df['trait_type'] == trait]['similarity_score'].values 
                   for trait in trait_types]
        bp = axes[0, 0].boxplot(box_data, labels=trait_types, showfliers=True, patch_artist=True)
        
        # Add red points for raw self-similarity values (should be 1.0)
        if 'raw_self_similarity' in top1_df.columns:
            for i, trait in enumerate(trait_types):
                trait_data = top1_df[top1_df['trait_type'] == trait]
                if len(trait_data) > 0:
                    # Get raw self-similarity values for this trait
                    raw_self_sim = trait_data['raw_self_similarity'].values
                    # Plot as red scatter points
                    x_positions = [i + 1] * len(raw_self_sim)  # boxplot positions are 1-indexed
                    axes[0, 0].scatter(x_positions, raw_self_sim, color='red', s=20, alpha=0.7, 
                                     label='Raw Self-Similarity' if i == 0 else "")
        
        axes[0, 0].set_title('Top-1 Similarity Scores by Trait Type')
        axes[0, 0].set_ylabel('Cosine Similarity')
        if 'raw_self_similarity' in top1_df.columns:
            axes[0, 0].legend()
        
        # Average similarity by rank
        rank_avg = results_df.groupby('rank')['similarity_score'].mean()
        axes[0, 1].plot(rank_avg.index, rank_avg.values, 'o-')
        axes[0, 1].set_title('Average Similarity Score by Rank')
        axes[0, 1].set_xlabel('Rank')
        axes[0, 1].set_ylabel('Average Cosine Similarity')
        
        # Trait type distribution
        trait_counts = results_df['trait_type'].value_counts()
        axes[1, 0].bar(trait_counts.index, trait_counts.values)
        axes[1, 0].set_title('Number of Analogy Tests by Trait Type')
        axes[1, 0].set_ylabel('Count')
        
        # Similarity score histogram
        axes[1, 1].hist(results_df['similarity_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of All Similarity Scores')
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('Frequency')
        
        # Self-match score vs prediction similarity comparison
        if 'self_match_score' in results_df.columns:
            # Scatter plot: self-match vs top-1 prediction similarity
            top1_with_self = results_df[results_df['rank'] == 1]
            axes[0, 2].scatter(top1_with_self['self_match_score'], top1_with_self['similarity_score'], 
                              alpha=0.6, s=30)
            axes[0, 2].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='y=x line')
            axes[0, 2].set_xlabel('Self-match Score')
            axes[0, 2].set_ylabel('Top-1 Prediction Similarity')
            axes[0, 2].set_title('Self-match vs Top-1 Prediction')
            axes[0, 2].legend()
            
            # Percentage of matches above self-match threshold
            above_self_pct = results_df.groupby('trait_type')['above_self_match'].mean() * 100
            axes[1, 2].bar(above_self_pct.index, above_self_pct.values)
            axes[1, 2].set_title('% Predictions Above Self-match by Trait')
            axes[1, 2].set_ylabel('Percentage')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"analogy_reasoning_analysis_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"analogy_reasoning_analysis_{self.timestamp}.pdf", bbox_inches='tight')
        plt.close()
        
        # Create high-quality matches specific visualization
        if 'above_self_match' in results_df.columns:
            self._create_high_quality_visualizations(results_df)
        
        logger.info("Visualizations saved")
    
    def _create_high_quality_visualizations(self, results_df: pd.DataFrame):
        """Create visualizations specifically for high-quality matches."""
        high_quality_df = results_df[results_df['above_self_match'] == True]
        
        if len(high_quality_df) == 0:
            logger.warning("No high-quality matches to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution of high-quality matches by trait type
        hq_trait_counts = high_quality_df['trait_type'].value_counts()
        axes[0, 0].bar(hq_trait_counts.index, hq_trait_counts.values, color='lightgreen')
        axes[0, 0].set_title('High-Quality Matches by Trait Type')
        axes[0, 0].set_ylabel('Count')
        
        # Distribution by node type
        node_types = {
            'NCBITaxon': len(high_quality_df[high_quality_df['predicted_taxon'].str.startswith('NCBITaxon:')]),
            'strain': len(high_quality_df[high_quality_df['predicted_taxon'].str.startswith('strain:')]),
            'pH nodes': len(high_quality_df[high_quality_df['predicted_taxon'].str.lower().str.startswith('ph_')]),
            'NaCl nodes': len(high_quality_df[high_quality_df['predicted_taxon'].str.lower().str.startswith('nacl_')])
        }
        axes[0, 1].bar(node_types.keys(), node_types.values(), color='lightcoral')
        axes[0, 1].set_title('High-Quality Matches by Node Type')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Similarity score distribution for high-quality matches
        axes[1, 0].hist(high_quality_df['similarity_score'], bins=30, alpha=0.7, 
                       color='green', edgecolor='black')
        axes[1, 0].set_title('Similarity Score Distribution (High-Quality)')
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('Frequency')
        
        # Rank distribution for high-quality matches
        rank_counts = high_quality_df['rank'].value_counts().sort_index()
        axes[1, 1].bar(rank_counts.index, rank_counts.values, color='orange')
        axes[1, 1].set_title('Rank Distribution (High-Quality)')
        axes[1, 1].set_xlabel('Prediction Rank')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"high_quality_matches_analysis_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"high_quality_matches_analysis_{self.timestamp}.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info(f"High-quality matches visualization saved ({len(high_quality_df)} matches)")


def main():
    """Main execution function."""
    logger.info("Starting Analogy Reasoning Analysis")
    
    # Initialize reasoner
    embeddings_path = "../output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-07-30_22_21_15.tsv.gz"
    reasoner = AnalogyReasoner(embeddings_path, data_dir="../")
    
    # Load data
    reasoner.load_embeddings()
    reasoner.load_trait_data_from_kg()
    
    # Run analysis
    results_df = reasoner.run_comprehensive_analysis()
    
    # Analyze results
    reasoner.analyze_results(results_df)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()