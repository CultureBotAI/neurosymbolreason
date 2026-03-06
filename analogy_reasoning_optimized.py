#!/usr/bin/env python3
"""
Optimized Analogy Reasoning on Node Embeddings for Microbial Physical Growth Preferences

This optimized version includes:
1. Batch cosine similarity calculations
2. Pre-computed trait vectors with caching
3. Parallel processing for multiple queries
4. Memory-efficient operations
5. Pre-filtered node indices for faster lookup

Author: Generated with Claude Code
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import gzip
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedAnalogyReasoner:
    """
    Optimized version of analogy reasoning with significant performance improvements.
    """
    
    def __init__(self, embeddings_path: str, data_dir: str = "../", output_dir: str = "./", 
                 use_float16: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the OptimizedAnalogyReasoner.
        
        Args:
            embeddings_path: Path to the DeepWalk embeddings file (.tsv.gz)
            data_dir: Directory containing trait data files  
            output_dir: Directory to save results
            use_float16: Use float16 to reduce memory usage (may reduce precision)
            max_workers: Number of parallel workers (default: 75% of CPU count)
        """
        self.embeddings_path = embeddings_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_float16 = use_float16
        # Detect if we're on a login node or HPC environment and limit workers accordingly
        if max_workers is None:
            total_cores = mp.cpu_count()
            # Use 12 workers or half of available cores, whichever is smaller
            self.max_workers = min(12, max(1, total_cores // 2))
            logger.info(f"Auto-detected {total_cores} cores. Using {self.max_workers} workers (max 12 or half of cores).")
        else:
            self.max_workers = max_workers
        
        # Create timestamp for output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data containers
        self.embeddings = {}
        self.embedding_matrix = None
        self.normalized_embeddings = None  # Pre-normalized for faster cosine similarity
        self.node_list = []
        self.node_to_index = {}  # Faster lookup
        
        # Pre-filtered indices for different node types
        self.ncbi_indices = []
        self.strain_indices = []
        self.ph_indices = []
        self.nacl_indices = []
        self.target_indices = []  # Combined target indices
        
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
        
        # Cached trait vectors
        self.trait_vectors_cache = {}
        
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
        """Load and preprocess DeepWalk embeddings with optimizations."""
        logger.info(f"Loading embeddings from {self.embeddings_path}")
        start_time = time.time()
        
        embeddings_dict = {}
        node_list = []
        
        dtype = np.float16 if self.use_float16 else np.float32
        
        with gzip.open(self.embeddings_path, 'rt') as f:
            # Skip header
            header = f.readline().strip().split('\t')
            embedding_dim = len(header) - 1
            logger.info(f"Embedding dimension: {embedding_dim}, using dtype: {dtype}")
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    logger.info(f"Loaded {line_num} embeddings...")
                    
                parts = line.strip().split('\t')
                node_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]], dtype=dtype)
                
                embeddings_dict[node_id] = embedding
                node_list.append(node_id)
        
        logger.info(f"Loaded {len(embeddings_dict)} embeddings in {time.time() - start_time:.2f}s")
        
        # Create optimized data structures
        start_time = time.time()
        self._create_optimized_structures(embeddings_dict, node_list)
        logger.info(f"Created optimized structures in {time.time() - start_time:.2f}s")
        
    def _create_optimized_structures(self, embeddings_dict: dict, node_list: list):
        """Create optimized data structures for fast lookup and computation."""
        # Create embedding matrix
        self.embedding_matrix = np.array([embeddings_dict[node] for node in node_list])
        self.embeddings = embeddings_dict
        self.node_list = node_list
        
        # Create node to index mapping for O(1) lookup
        self.node_to_index = {node: idx for idx, node in enumerate(node_list)}
        
        # Pre-normalize embeddings for faster cosine similarity (dot product of normalized vectors)
        self.normalized_embeddings = normalize(self.embedding_matrix, norm='l2', axis=1)
        
        # Pre-filter indices by node type for faster filtering
        for idx, node_id in enumerate(node_list):
            node_lower = node_id.lower()
            if node_id.startswith('NCBITaxon:'):
                self.ncbi_indices.append(idx)
                self.target_indices.append(idx)
            elif node_id.startswith('strain:'):
                self.strain_indices.append(idx)
                self.target_indices.append(idx)
            elif node_lower.startswith('ph_'):
                self.ph_indices.append(idx)
                self.target_indices.append(idx)
            elif node_lower.startswith('nacl_'):
                self.nacl_indices.append(idx)
                self.target_indices.append(idx)
        
        logger.info(f"Pre-filtered indices: NCBITaxon={len(self.ncbi_indices)}, "
                   f"strain={len(self.strain_indices)}, ph={len(self.ph_indices)}, "
                   f"nacl={len(self.nacl_indices)}, total_targets={len(self.target_indices)}")
    
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
        
        # Pre-compute and cache trait vectors
        self._precompute_trait_vectors()
    
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
        
        # Pre-compute and cache trait vectors
        self._precompute_trait_vectors()
    
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
    
    def _precompute_trait_vectors(self):
        """Pre-compute trait vectors for all trait types and values."""
        logger.info("Pre-computing trait vectors...")
        start_time = time.time()
        
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
            trait_vectors = defaultdict(list)
            
            for taxon, trait_value in trait_dict.items():
                if taxon in self.embeddings:
                    trait_vectors[trait_value].append(self.embeddings[taxon])
            
            # Compute mean vectors and cache them
            trait_means = {}
            for trait_value, vectors in trait_vectors.items():
                if len(vectors) > 0:
                    trait_means[trait_value] = np.mean(vectors, axis=0)
            
            self.trait_vectors_cache[trait_type] = trait_means
        
        logger.info(f"Pre-computed trait vectors in {time.time() - start_time:.2f}s")
    
    def get_taxa_with_traits(self) -> Dict[str, Dict[str, str]]:
        """Get all taxa that have at least one physical growth preference known."""
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
    
    def find_closest_nodes_optimized(self, target_vector: np.ndarray, top_k: int = 10, 
                                   exclude_query: str = None) -> List[Tuple[str, float]]:
        """
        Optimized version using pre-normalized embeddings and pre-filtered indices.
        """
        # Normalize target vector for cosine similarity via dot product
        target_normalized = normalize([target_vector], norm='l2')[0]
        
        # Compute dot products with all target nodes (equivalent to cosine similarity)
        target_embeddings = self.normalized_embeddings[self.target_indices]
        similarities = np.dot(target_embeddings, target_normalized)
        
        # Get top matches from target indices
        top_local_indices = np.argsort(similarities)[::-1][:top_k*2]  # Get extra for filtering
        
        results = []
        exclude_idx = self.node_to_index.get(exclude_query, -1)
        
        for local_idx in top_local_indices:
            global_idx = self.target_indices[local_idx]
            
            # Skip excluded query
            if global_idx == exclude_idx:
                continue
                
            node_id = self.node_list[global_idx]
            similarity = similarities[local_idx]
            results.append((node_id, float(similarity)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_analogy_self_match_score_optimized(self, query_taxon: str, predicted_vector: np.ndarray) -> float:
        """Optimized self-match score calculation."""
        if query_taxon not in self.node_to_index:
            return 0.0
        
        query_idx = self.node_to_index[query_taxon]
        query_vector = self.embedding_matrix[query_idx]
        
        # Use optimized cosine similarity
        query_norm = normalize([query_vector], norm='l2')[0]
        predicted_norm = normalize([predicted_vector], norm='l2')[0]
        similarity = np.dot(query_norm, predicted_norm)
        
        return float(similarity)
    
    def get_raw_self_similarity(self, query_taxon: str) -> float:
        """Get raw self-similarity score for a query taxon (should be 1.0)."""
        if query_taxon not in self.embeddings:
            return 0.0
        
        query_vector = self.embeddings[query_taxon]
        raw_self_similarity = cosine_similarity([query_vector], [query_vector])[0][0]
        return float(raw_self_similarity)
    
    def perform_analogy_reasoning_optimized(self, query_taxon: str, trait_type: str, 
                                          trait_value: str) -> Optional[Dict]:
        """Optimized analogy reasoning with cached trait vectors."""
        # Check if we have the necessary data
        if query_taxon not in self.embeddings:
            return None
            
        if trait_type not in self.trait_opposites:
            return None
            
        if trait_value not in self.trait_opposites[trait_type]:
            return None
        
        opposite_trait = self.trait_opposites[trait_type][trait_value]
        
        # Get cached trait vectors
        trait_cache = self.trait_vectors_cache.get(trait_type, {})
        
        if trait_value not in trait_cache or opposite_trait not in trait_cache:
            return None
            
        trait_vector = trait_cache[trait_value]
        opposite_vector = trait_cache[opposite_trait]
        
        # Perform analogy equation: predicted_vector = query_vector - trait_vector + opposite_trait_vector
        query_vector = self.embeddings[query_taxon]
        predicted_vector = query_vector - trait_vector + opposite_vector
        
        # Find closest matches using optimized method
        results = self.find_closest_nodes_optimized(
            predicted_vector, 
            top_k=10,
            exclude_query=query_taxon
        )
        
        # Get optimized self-match score
        self_match_score = self.get_analogy_self_match_score_optimized(query_taxon, predicted_vector)
        
        # Get raw self-similarity score
        raw_self_similarity = self.get_raw_self_similarity(query_taxon)
        
        return {
            'predictions': results,
            'self_match_score': self_match_score,
            'raw_self_similarity': raw_self_similarity
        }
    
    def process_query_batch(self, query_batch: List[Tuple[str, str, str]]) -> List[Dict]:
        """Process a batch of queries for parallel execution."""
        results = []
        for query_taxon, trait_type, trait_value in query_batch:
            result_data = self.perform_analogy_reasoning_optimized(query_taxon, trait_type, trait_value)
            if result_data:
                predictions = result_data['predictions']
                self_match_score = result_data['self_match_score']
                raw_self_similarity = result_data['raw_self_similarity']
                
                for rank, (predicted_taxon, similarity) in enumerate(predictions, 1):
                    result = {
                        'query_taxon': query_taxon,
                        'trait_type': trait_type,
                        'trait_value': trait_value,
                        'opposite_trait': self.trait_opposites[trait_type].get(trait_value, 'unknown'),
                        'rank': rank,
                        'predicted_taxon': predicted_taxon,
                        'similarity_score': similarity,
                        'self_match_score': self_match_score,
                        'raw_self_similarity': raw_self_similarity,
                        'above_self_match': similarity > self_match_score
                    }
                    results.append(result)
        return results
    
    def run_comprehensive_analysis_optimized(self):
        """Run optimized analogy reasoning with parallel processing."""
        logger.info("Starting optimized comprehensive analogy reasoning analysis...")
        start_time = time.time()
        
        # Get all taxa with traits
        taxa_traits = self.get_taxa_with_traits()
        
        # Create query list
        queries = []
        for taxon, traits in taxa_traits.items():
            for trait_type, trait_value in traits.items():
                queries.append((taxon, trait_type, trait_value))
        
        logger.info(f"Processing {len(queries)} queries with {self.max_workers} workers...")
        
        # Split queries into batches for parallel processing
        batch_size = max(1, len(queries) // (self.max_workers * 4))  # 4 batches per worker
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        
        all_results = []
        
        # Use ThreadPoolExecutor for I/O bound operations (since we're using numpy operations)
        logger.info(f"Created {len(query_batches)} batches (avg {batch_size} queries/batch) for parallel processing...")
        logger.info(f"Starting parallel batch processing with {self.max_workers} workers...")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches and track progress
            future_to_batch = {executor.submit(self.process_query_batch, batch): i 
                              for i, batch in enumerate(query_batches)}
            
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_result = future.result()
                all_results.extend(batch_result)
                completed_batches += 1
                
                # Log progress every 10% or every 5 batches, whichever is more frequent
                progress_interval = max(1, min(5, len(query_batches) // 10))
                if completed_batches % progress_interval == 0 or completed_batches == len(query_batches):
                    elapsed = time.time() - start_time
                    progress_pct = (completed_batches / len(query_batches)) * 100
                    queries_processed = sum(len(batch) for batch in list(query_batches)[:completed_batches])
                    queries_per_sec = queries_processed / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {completed_batches}/{len(query_batches)} batches ({progress_pct:.1f}%) - "
                               f"{queries_processed} queries processed - {queries_per_sec:.1f} queries/sec")
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(queries)} queries in {processing_time:.2f}s "
                   f"({len(queries)/processing_time:.1f} queries/sec)")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        output_file = self.output_dir / f"analogy_reasoning_results_optimized_{self.timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(all_results)} results to {output_file}")
        
        # Create and save high-quality matches analysis
        self._analyze_high_quality_matches(results_df)
        
        return results_df
    
    def _analyze_high_quality_matches(self, results_df: pd.DataFrame):
        """Analyze and save high-quality matches (reused from original implementation)."""
        logger.info("Analyzing high-quality matches...")
        
        enhanced_results = []
        
        for (query_taxon, trait_type, trait_value), group in results_df.groupby(['query_taxon', 'trait_type', 'trait_value']):
            self_match_score = group['self_match_score'].iloc[0]
            similarities = group['similarity_score'].values
            
            # Calculate statistical thresholds
            close_threshold = self_match_score * 0.95
            std_dev = np.std(similarities)
            
            for _, row in group.iterrows():
                similarity = row['similarity_score']
                above_self = similarity > self_match_score
                statistically_close = similarity > close_threshold
                
                enhanced_row = row.copy()
                enhanced_row['statistically_close_to_self'] = statistically_close
                enhanced_row['in_top_percentile'] = row['rank'] <= 3
                enhanced_row['similarity_std_dev'] = std_dev
                enhanced_row['close_threshold'] = close_threshold
                
                if above_self or statistically_close:
                    enhanced_results.append(enhanced_row)
        
        if enhanced_results:
            high_quality_detailed_df = pd.DataFrame(enhanced_results)
            
            # Save detailed high-quality results
            high_quality_file = self.output_dir / f"high_quality_matches_detailed_optimized_{self.timestamp}.csv"
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
            summary_file = self.output_dir / f"high_quality_matches_summary_optimized_{self.timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"High-quality match analysis complete. {summary_stats['queries_with_high_quality_matches']}/{summary_stats['total_queries']} queries had high-quality matches.")
        else:
            logger.warning("No high-quality matches found!")


def main():
    """Main execution function with optimization options."""
    logger.info("Starting Optimized Analogy Reasoning Analysis")
    
    # Initialize optimized reasoner
    embeddings_path = "../output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-07-30_22_21_15.tsv.gz"
    reasoner = OptimizedAnalogyReasoner(
        embeddings_path, 
        data_dir="../",
        use_float16=True,   # Use float16 to reduce memory usage on login node
        max_workers=None    # Auto-detect: uses 12 workers or half of cores (whichever is smaller)
    )
    
    # Load data
    reasoner.load_embeddings()
    reasoner.load_trait_data_from_kg()
    
    # Run optimized analysis
    results_df = reasoner.run_comprehensive_analysis_optimized()
    
    logger.info("Optimized analysis complete!")


if __name__ == "__main__":
    main()