#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from typing import Dict, List, Tuple

def load_and_process_data(results_dir):
    """Load results from a previous analysis"""
    # Load the numpy array of sink scores
    if os.path.exists(os.path.join(results_dir, 'avg_sink_scores.npy')):
        avg_sink_scores = np.load(os.path.join(results_dir, 'avg_sink_scores.npy'))
    else:
        raise FileNotFoundError(f"Cannot find avg_sink_scores.npy in {results_dir}")
    
    # Load the query results if available
    query_results = None
    if os.path.exists(os.path.join(results_dir, 'query_results.json')):
        with open(os.path.join(results_dir, 'query_results.json'), 'r') as f:
            query_results = json.load(f)
    
    return avg_sink_scores, query_results

def create_enhanced_visualizations(avg_sink_scores, query_results=None, output_dir="enhanced_visualizations", threshold=0.8):
    """Create enhanced visualizations focusing on sink patterns by layer"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    num_layers, num_heads = avg_sink_scores.shape
    
    # Calculate average sink score by layer
    avg_score_by_layer = avg_sink_scores.mean(axis=1)
    
    # Create a more detailed layer-wise analysis
    plt.figure(figsize=(15, 8))
    
    # Create bars
    bars = plt.bar(range(num_layers), avg_score_by_layer, color='skyblue', edgecolor='navy')
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Add the threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    # Enhance the visualization
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Average Attention to First Token', fontsize=14)
    plt.title('Average Attention to First Token by Layer', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_sink_score_by_layer_enhanced.png'), dpi=300)
    plt.close()
    
    # Create a heatmap with improved visibility
    plt.figure(figsize=(16, 10))
    
    # Use a more contrasting colormap
    heatmap = plt.imshow(avg_sink_scores, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Average Attention to First Token', fontsize=12)
    
    # Add grid
    plt.grid(False)
    
    # Add labels
    plt.xlabel('Head', fontsize=14)
    plt.ylabel('Layer', fontsize=14)
    plt.title('Attention Sink Patterns Across Layers and Heads', fontsize=16)
    
    # Add threshold contour
    contour_levels = [threshold]
    contour = plt.contour(np.arange(num_heads) + 0.5, np.arange(num_layers) + 0.5, 
                         avg_sink_scores, contour_levels, colors='red', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=10, fmt=f'Threshold: {threshold}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sink_heatmap_enhanced.png'), dpi=300)
    plt.close()
    
    # If we have query_results, visualize the trend of sink percentage by sequence length
    if query_results:
        seq_lengths = [result['seq_length'] for result in query_results]
        sink_percentages = [result['sink_percentage'] for result in query_results]
        
        # Sort by sequence length
        sorted_indices = np.argsort(seq_lengths)
        seq_lengths = [seq_lengths[i] for i in sorted_indices]
        sink_percentages = [sink_percentages[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        plt.plot(seq_lengths, sink_percentages, 'o-', linewidth=2, markersize=10)
        
        for i, (x, y) in enumerate(zip(seq_lengths, sink_percentages)):
            plt.text(x, y + 1, f'{y:.1f}%', ha='center')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Sequence Length', fontsize=14)
        plt.ylabel('Percentage of Attention Sink Heads', fontsize=14)
        plt.title('Relationship Between Sequence Length and Attention Sink Heads', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seq_length_vs_sink_percentage.png'), dpi=300)
        plt.close()
    
    # Create a visualization showing head consistency across layers
    # Calculate how consistent each head is across layers
    head_consistency = np.std(avg_sink_scores, axis=0)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(num_heads), head_consistency, color='lightgreen', edgecolor='darkgreen')
    
    plt.xlabel('Head', fontsize=14)
    plt.ylabel('Standard Deviation Across Layers', fontsize=14)
    plt.title('Head Consistency in Attention Patterns Across Layers', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'head_consistency.png'), dpi=300)
    plt.close()
    
    # Create a 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for 3D surface
    x = np.arange(num_heads)
    y = np.arange(num_layers)
    x, y = np.meshgrid(x, y)
    
    # Plot the surface
    surf = ax.plot_surface(x, y, avg_sink_scores, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add labels
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Attention to First Token', fontsize=12)
    ax.set_title('3D View of Attention Sink Patterns', fontsize=16)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add threshold plane
    xx, yy = np.meshgrid(np.arange(num_heads), np.arange(num_layers))
    ax.plot_surface(xx, yy, threshold * np.ones_like(xx), color='r', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_sink_patterns.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create enhanced visualizations of attention sink patterns")
    parser.add_argument("--results-dir", type=str, required=True, 
                        help="Directory containing analysis results")
    parser.add_argument("--output-dir", type=str, default="enhanced_visualizations",
                        help="Directory to save enhanced visualizations")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Threshold to classify as attention sink")
    
    args = parser.parse_args()
    
    # Load and process data
    avg_sink_scores, query_results = load_and_process_data(args.results_dir)
    
    # Create enhanced visualizations
    create_enhanced_visualizations(
        avg_sink_scores, 
        query_results, 
        args.output_dir,
        args.threshold
    )
    
    print(f"Enhanced visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()