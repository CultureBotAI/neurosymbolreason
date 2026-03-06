#!/usr/bin/env python3
"""
Ultra-fast Analogy Reasoning using Approximate Nearest Neighbors (ANN)

This version adds:
1. FAISS or Annoy for approximate nearest neighbor search
2. Quantization techniques for memory efficiency
3. Streaming processing for very large datasets
4. Advanced caching strategies

Author: Generated with Claude Code
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import gzip
from pathlib import Path
import logging
from sklearn.preprocessing import normalize
import time
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from datetime import datetime

# Optional imports for ultra-fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    logging.warning("Annoy not available. Install with: pip install annoy")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraFastAnalogyReasoner:
    """
    Ultra-optimized analogy reasoning with approximate nearest neighbor search.
    """
    
    def __init__(self, embeddings_path: str, data_dir: str = "../", output_dir: str = "./",
                 use_faiss: bool = True, use_float16: bool = False, 
                 ann_index_params: Optional[Dict] = None):
        """
        Initialize ultra-fast analogy reasoner.
        
        Args:
            embeddings_path: Path to embeddings file
            data_dir: Data directory
            output_dir: Output directory  
            use_faiss: Use FAISS for ANN search (requires faiss-cpu)
            use_float16: Use float16 for memory efficiency
            ann_index_params: Parameters for ANN index
        """
        self.embeddings_path = embeddings_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_float16 = use_float16
        self.ann_index_params = ann_index_params or {}
        
        # Create timestamp for output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data structures
        self.embeddings = {}
        self.embedding_matrix = None
        self.normalized_embeddings = None
        self.node_list = []
        self.node_to_index = {}
        
        # Target node filtering
        self.target_indices = []
        self.target_node_list = []
        
        # ANN indices
        self.faiss_index = None
        self.annoy_index = None
        
        # Trait data and caches
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
    
    def load_embeddings_streaming(self, chunk_size: int = 10000):
        """Load embeddings with streaming for memory efficiency."""
        logger.info(f"Loading embeddings with streaming (chunk_size={chunk_size})")
        start_time = time.time()
        
        dtype = np.float16 if self.use_float16 else np.float32
        embeddings_list = []
        node_list = []
        
        with gzip.open(self.embeddings_path, 'rt') as f:
            # Skip header and get dimension
            header = f.readline().strip().split('\t')
            embedding_dim = len(header) - 1
            
            chunk_embeddings = []
            chunk_nodes = []
            
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                node_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]], dtype=dtype)
                
                chunk_embeddings.append(embedding)
                chunk_nodes.append(node_id)
                
                # Process chunk
                if len(chunk_embeddings) >= chunk_size:
                    self._process_embedding_chunk(chunk_embeddings, chunk_nodes, 
                                                embeddings_list, node_list)
                    chunk_embeddings = []
                    chunk_nodes = []
                
                if line_num % 100000 == 0:
                    logger.info(f"Processed {line_num} embeddings...")
            
            # Process final chunk
            if chunk_embeddings:
                self._process_embedding_chunk(chunk_embeddings, chunk_nodes, 
                                            embeddings_list, node_list)
        
        # Create final structures
        self.embedding_matrix = np.vstack(embeddings_list) if embeddings_list else np.array([])
        self.node_list = node_list
        self.node_to_index = {node: idx for idx, node in enumerate(node_list)}
        self.embeddings = {node: self.embedding_matrix[idx] for idx, node in enumerate(node_list)}
        
        logger.info(f"Loaded {len(self.node_list)} embeddings in {time.time() - start_time:.2f}s")
        
        # Create optimized structures
        self._create_ultra_fast_structures()
    
    def _process_embedding_chunk(self, chunk_embeddings: List[np.ndarray], 
                                chunk_nodes: List[str], embeddings_list: List[np.ndarray], 
                                node_list: List[str]):
        """Process a chunk of embeddings."""
        chunk_matrix = np.array(chunk_embeddings)
        embeddings_list.append(chunk_matrix)
        node_list.extend(chunk_nodes)
    
    def _create_ultra_fast_structures(self):
        """Create ultra-fast data structures including ANN indices."""
        logger.info("Creating ultra-fast data structures...")
        start_time = time.time()
        
        # Filter target nodes and create target structures
        target_indices = []
        target_nodes = []
        
        for idx, node_id in enumerate(self.node_list):
            node_lower = node_id.lower()
            if (node_id.startswith('NCBITaxon:') or node_id.startswith('strain:') or 
                node_lower.startswith('ph_') or node_lower.startswith('nacl_')):
                target_indices.append(idx)
                target_nodes.append(node_id)
        
        self.target_indices = np.array(target_indices)
        self.target_node_list = target_nodes
        
        # Create target embedding matrix and normalize
        self.target_embeddings = self.embedding_matrix[self.target_indices]
        self.normalized_target_embeddings = normalize(self.target_embeddings, norm='l2', axis=1)
        
        # Normalize all embeddings for consistency
        self.normalized_embeddings = normalize(self.embedding_matrix, norm='l2', axis=1)
        
        logger.info(f"Target nodes: {len(self.target_indices)} out of {len(self.node_list)} total")
        
        # Build ANN index
        self._build_ann_index()
        
        logger.info(f"Created ultra-fast structures in {time.time() - start_time:.2f}s")
    
    def _build_ann_index(self):
        """Build approximate nearest neighbor index."""
        if not (FAISS_AVAILABLE or ANNOY_AVAILABLE):
            logger.warning("No ANN library available, falling back to exact search")
            return
        
        embedding_dim = self.target_embeddings.shape[1]
        n_vectors = len(self.target_embeddings)
        
        logger.info(f"Building ANN index for {n_vectors} vectors of dimension {embedding_dim}")
        start_time = time.time()
        
        if self.use_faiss and FAISS_AVAILABLE:
            # FAISS index
            index_type = self.ann_index_params.get('index_type', 'IVFFlat')
            n_clusters = self.ann_index_params.get('n_clusters', min(4096, n_vectors // 39))
            
            if index_type == 'IVFFlat':
                quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
                self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
                self.faiss_index.train(self.normalized_target_embeddings.astype(np.float32))
                self.faiss_index.add(self.normalized_target_embeddings.astype(np.float32))
                # Set search parameters
                self.faiss_index.nprobe = self.ann_index_params.get('nprobe', 32)
            else:
                # Fallback to flat index
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index.add(self.normalized_target_embeddings.astype(np.float32))
            
            logger.info(f"Built FAISS index in {time.time() - start_time:.2f}s")
            
        elif ANNOY_AVAILABLE:
            # Annoy index
            n_trees = self.ann_index_params.get('n_trees', 100)
            self.annoy_index = AnnoyIndex(embedding_dim, 'angular')  # Angular = cosine
            
            for i, vector in enumerate(self.normalized_target_embeddings):
                self.annoy_index.add_item(i, vector.astype(np.float32))
            
            self.annoy_index.build(n_trees)
            logger.info(f"Built Annoy index in {time.time() - start_time:.2f}s")
    
    def find_closest_nodes_ann(self, target_vector: np.ndarray, top_k: int = 10,
                              exclude_query: str = None) -> List[Tuple[str, float]]:
        """Find closest nodes using approximate nearest neighbor search."""
        # Normalize target vector
        target_normalized = normalize([target_vector], norm='l2')[0]
        
        exclude_idx = -1
        if exclude_query and exclude_query in self.node_to_index:
            global_exclude_idx = self.node_to_index[exclude_query]
            # Find if excluded node is in target indices
            target_positions = np.where(self.target_indices == global_exclude_idx)[0]
            if len(target_positions) > 0:
                exclude_idx = target_positions[0]
        
        if self.faiss_index is not None:
            # FAISS search
            search_k = min(top_k * 2, len(self.target_indices))  # Get extra for filtering
            similarities, indices = self.faiss_index.search(
                target_normalized.astype(np.float32).reshape(1, -1), search_k
            )
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == exclude_idx:
                    continue
                if idx >= 0 and idx < len(self.target_node_list):  # Valid index
                    node_id = self.target_node_list[idx]
                    results.append((node_id, float(similarity)))
                    if len(results) >= top_k:
                        break
            
            return results
            
        elif self.annoy_index is not None:
            # Annoy search
            search_k = min(top_k * 2, len(self.target_indices))
            indices = self.annoy_index.get_nns_by_vector(
                target_normalized.astype(np.float32), search_k, include_distances=False
            )
            
            results = []
            for idx in indices:
                if idx == exclude_idx:
                    continue
                if idx < len(self.target_node_list):
                    node_id = self.target_node_list[idx]
                    # Calculate exact similarity for ranking
                    similarity = np.dot(self.normalized_target_embeddings[idx], target_normalized)
                    results.append((node_id, float(similarity)))
                    if len(results) >= top_k:
                        break
            
            return results
        
        else:
            # Fallback to exact search
            similarities = np.dot(self.normalized_target_embeddings, target_normalized)
            top_indices = np.argsort(similarities)[::-1][:top_k*2]
            
            results = []
            for local_idx in top_indices:
                if local_idx == exclude_idx:
                    continue
                node_id = self.target_node_list[local_idx]
                similarity = similarities[local_idx]
                results.append((node_id, float(similarity)))
                if len(results) >= top_k:
                    break
            
            return results
    
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
        
        # Pre-compute trait vectors
        self._precompute_trait_vectors()
    
    def load_trait_data(self):
        """Load trait data and pre-compute vectors (fallback method)."""
        logger.info("Loading trait data from individual files...")
        
        # Load trait files (same as optimized version)
        oxygen_file = self.data_dir / "output" / "NCBITaxon_to_oxygen.tsv"
        if oxygen_file.exists():
            df = pd.read_csv(oxygen_file, sep='\t')
            self.oxygen_data = dict(zip(df['subject'], df['object'].str.replace('oxygen:', '')))
        
        salinity_file = self.data_dir / "taxa_media" / "NCBITaxon_to_salinity_v3.tsv"
        if salinity_file.exists():
            df = pd.read_csv(salinity_file, sep='\t')
            df_filtered = df[df['subject'].str.startswith('NCBITaxon:')]
            self.salinity_data = dict(zip(df_filtered['subject'], df_filtered['object'].str.replace('salinity:', '')))
        
        ph_file = self.data_dir / "taxa_media" / "taxa_pH_opt_mapping_adjusted_v2.tsv"
        if ph_file.exists():
            df = pd.read_csv(ph_file, sep='\t')
            self.ph_data = dict(zip(df['NCBITaxon'], df['pH_opt'].str.replace('pH_opt:', '')))
        
        temp_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_opt_v2.tsv"
        if temp_file.exists():
            df = pd.read_csv(temp_file, sep='\t')
            self.temp_data = dict(zip(df['subject'], df['object'].str.replace('temp_opt:', '')))
        
        # Load new trait data files (these may need to be created or paths adjusted)
        # pH range data
        ph_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_pH_range.tsv"
        if ph_range_file.exists():
            df = pd.read_csv(ph_range_file, sep='\t')
            self.ph_range_data = dict(zip(df['subject'], df['object'].str.replace('pH_range:', '')))
        
        # pH delta data
        ph_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_pH_delta.tsv"
        if ph_delta_file.exists():
            df = pd.read_csv(ph_delta_file, sep='\t')
            self.ph_delta_data = dict(zip(df['subject'], df['object'].str.replace('pH_delta:', '')))
        
        # NaCl range data
        nacl_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_range.tsv"
        if nacl_range_file.exists():
            df = pd.read_csv(nacl_range_file, sep='\t')
            self.nacl_range_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_range:', '')))
        
        # NaCl delta data
        nacl_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_delta.tsv"
        if nacl_delta_file.exists():
            df = pd.read_csv(nacl_delta_file, sep='\t')
            self.nacl_delta_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_delta:', '')))
        
        # NaCl optimal data
        nacl_opt_file = self.data_dir / "taxa_media" / "NCBITaxon_to_NaCl_opt.tsv"
        if nacl_opt_file.exists():
            df = pd.read_csv(nacl_opt_file, sep='\t')
            self.nacl_opt_data = dict(zip(df['subject'], df['object'].str.replace('NaCl_opt:', '')))
        
        # Motility data
        motility_file = self.data_dir / "taxa_media" / "NCBITaxon_to_motility.tsv"
        if motility_file.exists():
            df = pd.read_csv(motility_file, sep='\t')
            self.motility_data = dict(zip(df['subject'], df['object'].str.replace('motility:', '')))
        
        # Temperature range data
        temp_range_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_range.tsv"
        if temp_range_file.exists():
            df = pd.read_csv(temp_range_file, sep='\t')
            self.temp_range_data = dict(zip(df['subject'], df['object'].str.replace('temp_range:', '')))
        
        # Temperature delta data
        temp_delta_file = self.data_dir / "taxa_media" / "NCBITaxon_to_temp_delta.tsv"
        if temp_delta_file.exists():
            df = pd.read_csv(temp_delta_file, sep='\t')
            self.temp_delta_data = dict(zip(df['subject'], df['object'].str.replace('temp_delta:', '')))
        
        # Cell width data
        cell_width_file = self.data_dir / "taxa_media" / "NCBITaxon_to_cell_width.tsv"
        if cell_width_file.exists():
            df = pd.read_csv(cell_width_file, sep='\t')
            self.cell_width_data = dict(zip(df['subject'], df['object'].str.replace('cell_width:', '')))
        
        # Cell length data
        cell_length_file = self.data_dir / "taxa_media" / "NCBITaxon_to_cell_length.tsv"
        if cell_length_file.exists():
            df = pd.read_csv(cell_length_file, sep='\t')
            self.cell_length_data = dict(zip(df['subject'], df['object'].str.replace('cell_length:', '')))
        
        # Build dynamic trait opposites after all data is loaded
        self._build_dynamic_trait_opposites()
        
        # Pre-compute trait vectors
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
        """Pre-compute trait vectors."""
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
            
            trait_means = {}
            for trait_value, vectors in trait_vectors.items():
                if len(vectors) > 0:
                    trait_means[trait_value] = np.mean(vectors, axis=0)
            
            self.trait_vectors_cache[trait_type] = trait_means
    
    def get_raw_self_similarity(self, query_taxon: str) -> float:
        """Get raw self-similarity score for a query taxon (should be 1.0)."""
        if query_taxon not in self.embeddings:
            return 0.0
        
        query_vector = self.embeddings[query_taxon]
        # Use normalized vectors for consistency with cosine similarity
        query_norm = normalize([query_vector], norm='l2')[0]
        raw_self_similarity = np.dot(query_norm, query_norm)
        return float(raw_self_similarity)
    
    def perform_analogy_reasoning_ultra_fast(self, query_taxon: str, trait_type: str, 
                                           trait_value: str) -> Optional[Dict]:
        """Ultra-fast analogy reasoning."""
        if (query_taxon not in self.embeddings or 
            trait_type not in self.trait_opposites or
            trait_value not in self.trait_opposites[trait_type]):
            return None
        
        opposite_trait = self.trait_opposites[trait_type][trait_value]
        trait_cache = self.trait_vectors_cache.get(trait_type, {})
        
        if trait_value not in trait_cache or opposite_trait not in trait_cache:
            return None
        
        # Perform analogy
        query_vector = self.embeddings[query_taxon]
        trait_vector = trait_cache[trait_value]
        opposite_vector = trait_cache[opposite_trait]
        predicted_vector = query_vector - trait_vector + opposite_vector
        
        # Find matches using ANN
        results = self.find_closest_nodes_ann(predicted_vector, top_k=10, exclude_query=query_taxon)
        
        # Calculate self-match score
        query_norm = normalize([query_vector], norm='l2')[0]
        predicted_norm = normalize([predicted_vector], norm='l2')[0]
        self_match_score = float(np.dot(query_norm, predicted_norm))
        
        # Get raw self-similarity score
        raw_self_similarity = self.get_raw_self_similarity(query_taxon)
        
        return {
            'predictions': results,
            'self_match_score': self_match_score,
            'raw_self_similarity': raw_self_similarity
        }
    
    def benchmark_performance(self, n_queries: int = 100):
        """Benchmark the ultra-fast implementation."""
        logger.info(f"Benchmarking performance with {n_queries} queries...")
        
        # Get sample queries
        taxa_traits = {}
        for taxon, traits in [(t, {'oxygen': v}) for t, v in list(self.oxygen_data.items())[:n_queries]]:
            if taxon in self.embeddings:
                taxa_traits[taxon] = traits
                if len(taxa_traits) >= n_queries:
                    break
        
        queries = [(taxon, trait_type, trait_value) 
                  for taxon, traits in taxa_traits.items() 
                  for trait_type, trait_value in traits.items()][:n_queries]
        
        # Benchmark
        start_time = time.time()
        results = []
        
        for query_taxon, trait_type, trait_value in queries:
            result = self.perform_analogy_reasoning_ultra_fast(query_taxon, trait_type, trait_value)
            if result:
                results.append(result)
        
        total_time = time.time() - start_time
        queries_per_sec = len(queries) / total_time if total_time > 0 else 0
        
        logger.info(f"Processed {len(queries)} queries in {total_time:.3f}s")
        logger.info(f"Performance: {queries_per_sec:.1f} queries/second")
        
        benchmark_results = {
            'total_time': total_time,
            'queries_per_second': queries_per_sec,
            'successful_queries': len(results),
            'timestamp': self.timestamp,
            'configuration': {
                'use_faiss': self.use_faiss,
                'use_float16': self.use_float16,
                'ann_index_params': self.ann_index_params
            }
        }
        
        # Save benchmark results
        benchmark_file = self.output_dir / f"ultra_fast_benchmark_results_{self.timestamp}.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        logger.info(f"Saved benchmark results to {benchmark_file}")
        
        return benchmark_results
    
    def run_comprehensive_analysis_ultra_fast(self):
        """Run comprehensive analysis with ultra-fast optimizations and save results."""
        logger.info("Starting ultra-fast comprehensive analogy reasoning analysis...")
        start_time = time.time()
        
        # Get all taxa with traits (comprehensive version)
        taxa_traits = {}
        
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
        
        for trait_type, trait_dict in trait_data_map.items():
            for taxon, trait_value in trait_dict.items():
                if taxon in self.embeddings:
                    if taxon not in taxa_traits:
                        taxa_traits[taxon] = {}
                    taxa_traits[taxon][trait_type] = trait_value
        
        # Create query list
        queries = []
        for taxon, traits in taxa_traits.items():
            for trait_type, trait_value in traits.items():
                queries.append((taxon, trait_type, trait_value))
        
        logger.info(f"Processing {len(queries)} queries...")
        
        # Process queries
        all_results = []
        for query_taxon, trait_type, trait_value in queries:
            result_data = self.perform_analogy_reasoning_ultra_fast(query_taxon, trait_type, trait_value)
            
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
                    all_results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(queries)} queries in {processing_time:.2f}s "
                   f"({len(queries)/processing_time:.1f} queries/sec)")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        output_file = self.output_dir / f"analogy_reasoning_results_ultra_fast_{self.timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(all_results)} results to {output_file}")
        
        # Create summary statistics
        summary_stats = {
            'total_queries': len(results_df['query_taxon'].unique()) if len(results_df) > 0 else 0,
            'total_predictions': len(results_df),
            'processing_time_seconds': processing_time,
            'queries_per_second': len(queries)/processing_time if processing_time > 0 else 0,
            'avg_similarity_score': results_df['similarity_score'].mean() if len(results_df) > 0 else 0,
            'avg_self_match_score': results_df['self_match_score'].mean() if len(results_df) > 0 else 0,
            'matches_above_self_match': len(results_df[results_df['above_self_match']]) if len(results_df) > 0 else 0,
            'timestamp': self.timestamp,
            'configuration': {
                'use_faiss': self.use_faiss,
                'use_float16': self.use_float16,
                'ann_index_params': self.ann_index_params
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"ultra_fast_analysis_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        logger.info(f"Saved analysis summary to {summary_file}")
        
        return results_df

def main():
    """Main execution with ultra-fast optimizations."""
    logger.info("Starting Ultra-Fast Analogy Reasoning Analysis")
    
    # Initialize ultra-fast reasoner
    embeddings_path = "../output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-07-30_22_21_15.tsv.gz"
    
    # Configure ANN parameters
    ann_params = {
        'index_type': 'IVFFlat',  # FAISS index type
        'n_clusters': 2048,       # Number of clusters for IVF
        'nprobe': 32,            # Search parameter
        'n_trees': 100           # Annoy trees
    }
    
    reasoner = UltraFastAnalogyReasoner(
        embeddings_path,
        data_dir="../",
        use_faiss=True,
        use_float16=False,
        ann_index_params=ann_params
    )
    
    # Load data with streaming
    reasoner.load_embeddings_streaming(chunk_size=10000)
    reasoner.load_trait_data_from_kg()
    
    # Benchmark performance
    performance = reasoner.benchmark_performance(n_queries=1000)
    
    logger.info("Ultra-fast analysis complete!")
    logger.info(f"Final performance: {performance['queries_per_second']:.1f} queries/second")

if __name__ == "__main__":
    main()