#!/usr/bin/env python3
"""
Performance benchmark script to compare original vs optimized analogy reasoning.

This script tests various optimization strategies and measures their impact.
"""

import time
import numpy as np
import psutil
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Benchmark different optimization strategies."""
    
    def __init__(self, n_vectors: int = 100000, embedding_dim: int = 500):
        """Initialize benchmark with test data."""
        self.n_vectors = n_vectors
        self.embedding_dim = embedding_dim
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate test data
        logger.info(f"Generating {n_vectors} test vectors of dimension {embedding_dim}")
        self.embeddings = np.random.randn(n_vectors, embedding_dim).astype(np.float32)
        self.query_vector = np.random.randn(embedding_dim).astype(np.float32)
        
        # Pre-compute normalized versions
        self.normalized_embeddings = normalize(self.embeddings, norm='l2', axis=1)
        self.normalized_query = normalize([self.query_vector], norm='l2')[0]
        
        # Create some indices for filtering tests
        self.target_indices = np.random.choice(n_vectors, size=n_vectors//4, replace=False)
        
    def benchmark_cosine_similarity_methods(self):
        """Benchmark different cosine similarity calculation methods."""
        logger.info("Benchmarking cosine similarity methods...")
        
        methods = {}
        
        # Method 1: sklearn cosine_similarity (original)
        start_time = time.time()
        similarities_sklearn = cosine_similarity([self.query_vector], self.embeddings)[0]
        methods['sklearn_original'] = time.time() - start_time
        
        # Method 2: Pre-normalized dot product
        start_time = time.time()
        similarities_normalized = np.dot(self.normalized_embeddings, self.normalized_query)
        methods['normalized_dot_product'] = time.time() - start_time
        
        # Method 3: Manual cosine calculation
        start_time = time.time()
        query_norm = np.linalg.norm(self.query_vector)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        dot_products = np.dot(self.embeddings, self.query_vector)
        similarities_manual = dot_products / (embedding_norms * query_norm)
        methods['manual_calculation'] = time.time() - start_time
        
        # Method 4: Batch normalized (most efficient)
        start_time = time.time()
        similarities_batch = np.dot(self.normalized_embeddings, self.normalized_query)
        methods['batch_normalized'] = time.time() - start_time
        
        # Verify results are similar
        logger.info("Verifying result similarity...")
        assert np.allclose(similarities_sklearn, similarities_normalized, atol=1e-6), "Normalized method differs"
        assert np.allclose(similarities_sklearn, similarities_manual, atol=1e-6), "Manual method differs"
        
        # Report results
        baseline = methods['sklearn_original']
        logger.info("Cosine similarity method comparison:")
        for method, time_taken in methods.items():
            speedup = baseline / time_taken if time_taken > 0 else float('inf')
            logger.info(f"  {method}: {time_taken:.4f}s (speedup: {speedup:.2f}x)")
        
        return methods
    
    def benchmark_filtering_methods(self):
        """Benchmark different filtering approaches."""
        logger.info("Benchmarking filtering methods...")
        
        methods = {}
        top_k = 10
        
        # Compute similarities first
        similarities = np.dot(self.normalized_embeddings, self.normalized_query)
        
        # Method 1: Original approach (sort all then filter)
        start_time = time.time()
        top_indices = np.argsort(similarities)[::-1][:top_k*5]
        filtered_results = []
        for idx in top_indices:
            if idx in self.target_indices:  # Simulate filtering condition
                filtered_results.append((idx, similarities[idx]))
                if len(filtered_results) >= top_k:
                    break
        methods['sort_then_filter'] = time.time() - start_time
        
        # Method 2: Pre-filter then sort (optimized)
        start_time = time.time()
        target_similarities = similarities[self.target_indices]
        top_local_indices = np.argsort(target_similarities)[::-1][:top_k]
        filtered_results_opt = [(self.target_indices[local_idx], target_similarities[local_idx]) 
                               for local_idx in top_local_indices]
        methods['filter_then_sort'] = time.time() - start_time
        
        # Method 3: Using np.argpartition (for large k)
        start_time = time.time()
        target_similarities = similarities[self.target_indices]
        # argpartition is faster for finding top-k when k << n
        partition_indices = np.argpartition(target_similarities, -top_k)[-top_k:]
        sorted_partition = partition_indices[np.argsort(target_similarities[partition_indices])[::-1]]
        filtered_results_partition = [(self.target_indices[local_idx], target_similarities[local_idx]) 
                                     for local_idx in sorted_partition]
        methods['argpartition'] = time.time() - start_time
        
        # Report results
        baseline = methods['sort_then_filter']
        logger.info("Filtering method comparison:")
        for method, time_taken in methods.items():
            speedup = baseline / time_taken if time_taken > 0 else float('inf')
            logger.info(f"  {method}: {time_taken:.4f}s (speedup: {speedup:.2f}x)")
        
        return methods
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage of different approaches."""
        logger.info("Benchmarking memory usage...")
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_usage = {'baseline': baseline_memory}
        
        # Test float32 vs float16
        embeddings_f16 = self.embeddings.astype(np.float16)
        f16_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['float16'] = f16_memory - baseline_memory
        
        # Test pre-normalized storage
        normalized_copy = self.normalized_embeddings.copy()
        normalized_memory = process.memory_info().rss / 1024 / 1024
        memory_usage['normalized_copy'] = normalized_memory - f16_memory
        
        logger.info("Memory usage comparison:")
        for usage_type, memory_mb in memory_usage.items():
            logger.info(f"  {usage_type}: {memory_mb:.1f} MB")
        
        # Clean up
        del embeddings_f16, normalized_copy
        
        return memory_usage
    
    def benchmark_parallel_processing(self):
        """Benchmark parallel processing approaches."""
        logger.info("Benchmarking parallel processing...")
        
        # Simulate multiple queries
        n_queries = 100
        query_vectors = np.random.randn(n_queries, self.embedding_dim).astype(np.float32)
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for query in query_vectors:
            query_norm = normalize([query], norm='l2')[0]
            similarities = np.dot(self.normalized_embeddings, query_norm)
            top_idx = np.argmax(similarities)
            sequential_results.append((top_idx, similarities[top_idx]))
        sequential_time = time.time() - start_time
        
        # Batch processing
        start_time = time.time()
        normalized_queries = normalize(query_vectors, norm='l2', axis=1)
        # Compute all similarities at once
        all_similarities = np.dot(self.normalized_embeddings, normalized_queries.T)
        top_indices = np.argmax(all_similarities, axis=0)
        batch_results = [(idx, all_similarities[idx, i]) for i, idx in enumerate(top_indices)]
        batch_time = time.time() - start_time
        
        speedup = sequential_time / batch_time if batch_time > 0 else float('inf')
        logger.info(f"Parallel processing comparison:")
        logger.info(f"  Sequential: {sequential_time:.4f}s")
        logger.info(f"  Batch: {batch_time:.4f}s (speedup: {speedup:.2f}x)")
        
        return {'sequential': sequential_time, 'batch': batch_time}
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks and create summary."""
        logger.info("Running comprehensive performance benchmark...")
        
        results = {}
        
        # Run individual benchmarks
        results['cosine_similarity'] = self.benchmark_cosine_similarity_methods()
        results['filtering'] = self.benchmark_filtering_methods()
        results['memory'] = self.benchmark_memory_usage()
        results['parallel'] = self.benchmark_parallel_processing()
        
        # Create summary visualization
        self.create_benchmark_visualization(results)
        
        # Save results to JSON
        self.save_benchmark_results(results)
        
        return results
    
    def create_benchmark_visualization(self, results):
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cosine similarity methods
        methods = list(results['cosine_similarity'].keys())
        times = list(results['cosine_similarity'].values())
        baseline = times[0]
        speedups = [baseline / t for t in times]
        
        axes[0, 0].bar(methods, speedups)
        axes[0, 0].set_title('Cosine Similarity Method Speedups')
        axes[0, 0].set_ylabel('Speedup (x)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Filtering methods
        methods = list(results['filtering'].keys())
        times = list(results['filtering'].values())
        baseline = times[0]
        speedups = [baseline / t for t in times]
        
        axes[0, 1].bar(methods, speedups)
        axes[0, 1].set_title('Filtering Method Speedups')
        axes[0, 1].set_ylabel('Speedup (x)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory_types = [k for k in results['memory'].keys() if k != 'baseline']
        memory_values = [results['memory'][k] for k in memory_types]
        
        axes[1, 0].bar(memory_types, memory_values)
        axes[1, 0].set_title('Additional Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Parallel processing
        methods = list(results['parallel'].keys())
        times = list(results['parallel'].values())
        
        axes[1, 1].bar(methods, times)
        axes[1, 1].set_title('Processing Time Comparison')
        axes[1, 1].set_ylabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(f'performance_benchmark_results_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'performance_benchmark_results_{self.timestamp}.pdf', bbox_inches='tight')
        logger.info("Benchmark visualization saved")
        
        plt.show()
    
    def save_benchmark_results(self, results):
        """Save detailed benchmark results to JSON."""
        detailed_results = {
            'timestamp': self.timestamp,
            'test_configuration': {
                'n_vectors': self.n_vectors,
                'embedding_dim': self.embedding_dim
            },
            'results': results,
            'performance_summary': {
                'cosine_similarity_best_speedup': max([
                    results['cosine_similarity']['sklearn_original'] / time_val 
                    for time_val in results['cosine_similarity'].values()
                ]),
                'filtering_best_speedup': max([
                    results['filtering']['sort_then_filter'] / time_val 
                    for time_val in results['filtering'].values()
                ]),
                'parallel_speedup': results['parallel']['sequential'] / results['parallel']['batch']
            }
        }
        
        with open(f'performance_benchmark_detailed_{self.timestamp}.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        logger.info(f"Detailed benchmark results saved to performance_benchmark_detailed_{self.timestamp}.json")

def main():
    """Run performance benchmarks."""
    logger.info("Starting performance benchmarks...")
    
    # Initialize benchmark with realistic data size
    benchmark = PerformanceBenchmark(n_vectors=100000, embedding_dim=500)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    logger.info("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    logger.info("1. Use pre-normalized embeddings with dot product for cosine similarity")
    logger.info("2. Pre-filter target indices instead of post-filtering results")
    logger.info("3. Use batch processing for multiple queries")
    logger.info("4. Consider float16 for memory-constrained environments")
    logger.info("5. Cache frequently used computations (trait vectors)")
    
    return results

if __name__ == "__main__":
    main()