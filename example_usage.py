#!/usr/bin/env python3
"""
Example usage script for analogy reasoning on microbial knowledge graph embeddings.

This script demonstrates how to use the AnalogyReasoner class for specific queries.
"""

from analogy_reasoning import AnalogyReasoner
import logging

# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_example_queries():
    """Run example analogy reasoning queries."""
    
    # Initialize reasoner
    embeddings_path = "../output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-04-07_03_18_35.tsv.gz"
    reasoner = AnalogyReasoner(embeddings_path, data_dir="../")
    
    # Load data
    logger.info("Loading embeddings and trait data...")
    reasoner.load_embeddings()  
    reasoner.load_trait_data()
    
    # Get taxa with traits for examples
    taxa_traits = reasoner.get_taxa_with_traits()
    
    # Example 1: Oxygen requirement analogy
    logger.info("\n=== Example 1: Oxygen Requirement Analogy ===")
    example_queries = [
        # Find taxa with oxygen preferences
        (taxon, 'oxygen', trait_val) for taxon, traits in taxa_traits.items() 
        if 'oxygen' in traits for trait_val in [traits['oxygen']]
    ][:3]  # Take first 3
    
    for query_taxon, trait_type, trait_value in example_queries:
        logger.info(f"\nQuery: {query_taxon} with {trait_type}:{trait_value}")
        logger.info(f"Analogy: {query_taxon} - {trait_value} + {reasoner.trait_opposites[trait_type].get(trait_value, 'unknown')}")
        
        result_data = reasoner.perform_analogy_reasoning(query_taxon, trait_type, trait_value)
        
        if result_data:
            predictions = result_data['predictions']
            self_match_score = result_data['self_match_score']
            
            logger.info(f"Self-match score: {self_match_score:.3f} (original taxon vs analogy result)")
            logger.info("Top 5 predictions (similarity to analogy result vector):")
            for rank, (predicted_taxon, similarity) in enumerate(predictions[:5], 1):
                above_self = "✓" if similarity > self_match_score else " "
                logger.info(f"  {rank}. {predicted_taxon} (similarity: {similarity:.3f}) {above_self}")
        else:
            logger.info("  No predictions generated")
    
    # Example 2: Temperature preference analogy
    logger.info("\n=== Example 2: Temperature Preference Analogy ===") 
    temp_queries = [
        (taxon, 'temp_opt', trait_val) for taxon, traits in taxa_traits.items()
        if 'temp_opt' in traits for trait_val in [traits['temp_opt']]
    ][:2]  # Take first 2
    
    for query_taxon, trait_type, trait_value in temp_queries:
        logger.info(f"\nQuery: {query_taxon} with {trait_type}:{trait_value}")
        logger.info(f"Analogy: {query_taxon} - {trait_value} + {reasoner.trait_opposites[trait_type].get(trait_value, 'unknown')}")
        
        result_data = reasoner.perform_analogy_reasoning(query_taxon, trait_type, trait_value)
        
        if result_data:
            predictions = result_data['predictions']
            self_match_score = result_data['self_match_score']
            
            logger.info(f"Self-match score: {self_match_score:.3f} (original taxon vs analogy result)")
            logger.info("Top 3 predictions (similarity to analogy result vector):")
            for rank, (predicted_taxon, similarity) in enumerate(predictions[:3], 1):
                above_self = "✓" if similarity > self_match_score else " "
                logger.info(f"  {rank}. {predicted_taxon} (similarity: {similarity:.3f}) {above_self}")
        else:
            logger.info("  No predictions generated")
    
    # Example 3: pH preference analogy
    logger.info("\n=== Example 3: pH Preference Analogy ===")
    ph_queries = [
        (taxon, 'ph_opt', trait_val) for taxon, traits in taxa_traits.items()
        if 'ph_opt' in traits for trait_val in [traits['ph_opt']]
    ][:2]  # Take first 2
    
    for query_taxon, trait_type, trait_value in ph_queries:
        logger.info(f"\nQuery: {query_taxon} with {trait_type}:{trait_value}")
        logger.info(f"Analogy: {query_taxon} - {trait_value} + {reasoner.trait_opposites[trait_type].get(trait_value, 'unknown')}")
        
        result_data = reasoner.perform_analogy_reasoning(query_taxon, trait_type, trait_value)
        
        if result_data:
            predictions = result_data['predictions']
            self_match_score = result_data['self_match_score']
            
            logger.info(f"Self-match score: {self_match_score:.3f} (original taxon vs analogy result)")
            logger.info("Top 3 predictions (similarity to analogy result vector):")
            for rank, (predicted_taxon, similarity) in enumerate(predictions[:3], 1):
                above_self = "✓" if similarity > self_match_score else " "
                logger.info(f"  {rank}. {predicted_taxon} (similarity: {similarity:.3f}) {above_self}")
        else:
            logger.info("  No predictions generated")

def demonstrate_manual_query():
    """Demonstrate how to run a specific manual query."""
    logger.info("\n=== Manual Query Example ===")
    
    # Initialize reasoner
    embeddings_path = "../output/DeepWalkSkipGramEnsmallen_degreenorm_embedding_500_2025-04-07_03_18_35.tsv.gz"
    reasoner = AnalogyReasoner(embeddings_path, data_dir="../")
    
    # Load data
    reasoner.load_embeddings()
    reasoner.load_trait_data()
    
    # Manual query - if you know a specific taxon ID
    specific_taxon = "NCBITaxon:562"  # E. coli (example)
    
    # Check if this taxon has embeddings and traits
    if specific_taxon in reasoner.embeddings:
        logger.info(f"Taxon {specific_taxon} found in embeddings")
        
        # Check oxygen preference
        if specific_taxon in reasoner.oxygen_data:
            oxygen_pref = reasoner.oxygen_data[specific_taxon]
            logger.info(f"Oxygen preference: {oxygen_pref}")
            
            # Run analogy
            result_data = reasoner.perform_analogy_reasoning(specific_taxon, 'oxygen', oxygen_pref)
            if result_data:
                predictions = result_data['predictions']
                self_match_score = result_data['self_match_score']
                
                logger.info(f"Analogy: {specific_taxon} - {oxygen_pref} + {reasoner.trait_opposites['oxygen'].get(oxygen_pref, 'unknown')}")
                logger.info(f"Self-match score: {self_match_score:.3f} (original taxon vs analogy result)")
                logger.info("Predictions (similarity to analogy result vector):")
                for rank, (pred_taxon, sim) in enumerate(predictions[:5], 1):
                    above_self = "✓" if sim > self_match_score else " "
                    logger.info(f"  {rank}. {pred_taxon} (similarity: {sim:.3f}) {above_self}")
        else:
            logger.info(f"No oxygen preference data for {specific_taxon}")
    else:
        logger.info(f"Taxon {specific_taxon} not found in embeddings")

if __name__ == "__main__":
    logger.info("Starting example analogy reasoning queries...")
    
    # Run example queries
    run_example_queries()
    
    # Demonstrate manual query
    demonstrate_manual_query()
    
    logger.info("\nExample queries complete!")