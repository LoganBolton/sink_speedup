#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from typing import Dict, List, Tuple

def simulate_attention_patterns(
    num_layers=32,  # Llama-3.1-8B has 32 layers
    num_heads=32,   # Llama-3.1-8B has 32 attention heads per layer
    num_queries=5,
    sequence_lengths=[5, 10, 15, 20, 25],
    output_dir="simulated_8b_results"
):
    """
    Simulate attention patterns for a large language model like Llama-3.1-8B.
    
    This function uses empirical observations about attention patterns to generate
    synthetic data that mimics real model behavior:
    
    1. Earlier layers have more attention sinks
    2. Shorter sequences have more attention sinks
    3. Some heads consistently behave as attention sinks
    4. Sink behavior decays as sequence length increases
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all simulated query results
    all_results = []
    
    # Sample 5 different prompts with varying lengths
    prompts = [
        f"Simulated prompt of length {seq_len}" 
        for seq_len in sequence_lengths
    ]
    
    # Create base sink probability by layer (decreasing as layer increases)
    # Earlier layers have higher probability of being sinks
    layer_sink_base_probs = np.linspace(0.9, 0.2, num_layers)
    
    # Create head sink consistency (some heads are more likely to be sinks across all layers)
    head_sink_tendency = np.random.beta(2, 3, size=num_heads)  # Random tendency for each head
    
    # Process each prompt
    for i, (prompt, seq_len) in enumerate(zip(prompts, sequence_lengths)):
        print(f"\nSimulating prompt {i+1}/{len(prompts)}: '{prompt}', length: {seq_len}")
        
        # Store layer-head sink scores for this query
        layer_head_sink_score = []
        
        # Sequence length effect - longer sequences decrease sink probability
        seq_length_factor = max(0.2, 1.0 - 0.03 * seq_len)  # Sequence length affects sink probability
        
        # For each layer
        for layer_idx in range(num_layers):
            layer_sink_scores = []
            
            # Base probability of this layer having sink heads
            layer_prob = layer_sink_base_probs[layer_idx] * seq_length_factor
            
            # For each head
            for head_idx in range(num_heads):
                # Head's consistent tendency + layer probability + some randomness
                head_tendency = head_sink_tendency[head_idx]
                sink_prob = layer_prob * head_tendency
                
                # Add some noise to the probability
                noise = np.random.normal(0, 0.05)
                sink_prob = max(0.05, min(0.98, sink_prob + noise))
                
                # For attention sinks, scores are mostly between 0.9-1.0, with some varying
                if np.random.random() < sink_prob:
                    sink_score = np.random.beta(15, 2)  # Mostly high scores (>0.9)
                else:
                    sink_score = np.random.beta(2, 5)  # Mostly low scores (<0.3)
                
                layer_sink_scores.append(float(sink_score))
            
            layer_head_sink_score.append(layer_sink_scores)
        
        # Calculate the sink percentage based on threshold
        total_heads = num_layers * num_heads
        sink_heads = sum(sum(1 for score in layer if score >= 0.9) for layer in layer_head_sink_score)
        sink_percentage = sink_heads / total_heads * 100
        
        # Store results for this query
        query_result = {
            "prompt": prompt,
            "seq_length": seq_len,
            "sink_percentage": sink_percentage,
            "layer_head_sink_score": layer_head_sink_score
        }
        all_results.append(query_result)
        
        print(f"Simulated sink heads: {sink_heads}/{total_heads} ({sink_percentage:.2f}%)")
    
    return all_results

def average_results(all_results):
    """Calculate average sink scores across all simulated queries"""
    # Get dimensions from first result
    first_result = all_results[0]
    num_layers = len(first_result["layer_head_sink_score"])
    num_heads = len(first_result["layer_head_sink_score"][0])
    
    # Initialize average scores
    avg_sink_scores = np.zeros((num_layers, num_heads))
    
    # Add up all scores
    for result in all_results:
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                avg_sink_scores[layer_idx, head_idx] += result["layer_head_sink_score"][layer_idx][head_idx]
    
    # Divide by number of queries
    avg_sink_scores /= len(all_results)
    
    return avg_sink_scores

def visualize_sink_frequency(avg_sink_scores, threshold=0.9, output_dir="simulated_8b_results"):
    """Create visualizations of sink frequencies"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a heatmap of average sink scores
    plt.figure(figsize=(14, 10))
    plt.imshow(avg_sink_scores, cmap="viridis", aspect="auto")
    plt.colorbar(label="Average proportion of attention on first token")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Average Attention Sink Scores (Simulated Llama-3.1-8B)\nthreshold={threshold}")
    
    # Add grid lines
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_sink_scores_heatmap.png", dpi=300)
    plt.close()
    
    # Create a mask of which heads exceed the threshold on average
    sink_mask = avg_sink_scores >= threshold
    
    plt.figure(figsize=(14, 10))
    plt.imshow(sink_mask, cmap="binary", aspect="auto")
    plt.colorbar(label="Is attention sink")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention Sink Mask (Simulated Llama-3.1-8B)\nthreshold={threshold}")
    
    # Add grid lines
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sink_mask_heatmap.png", dpi=300)
    plt.close()
    
    # Calculate and plot sink frequency by layer
    sink_by_layer = sink_mask.mean(axis=1) * 100  # % of heads that are sinks in each layer
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(sink_by_layer)), sink_by_layer)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel("Layer")
    plt.ylabel("Percentage of Heads that are Attention Sinks")
    plt.title("Frequency of Attention Sinks by Layer (Simulated Llama-3.1-8B)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sink_frequency_by_layer.png", dpi=300)
    plt.close()
    
    # Calculate and plot average sink score by layer
    avg_score_by_layer = avg_sink_scores.mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(avg_score_by_layer)), avg_score_by_layer)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel("Layer")
    plt.ylabel("Average Attention Sink Score")
    plt.title("Average Attention Sink Score by Layer (Simulated Llama-3.1-8B)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_sink_score_by_layer.png", dpi=300)
    plt.close()
    
    # Save the numerical data
    np.save(f"{output_dir}/avg_sink_scores.npy", avg_sink_scores)
    
    # Save a summary report
    sink_percentage = sink_mask.mean() * 100
    
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"Simulated Average Sink Analysis for Llama-3.1-8B\n\n")
        f.write(f"Total percentage of attention sink heads: {sink_percentage:.2f}%\n\n")
        f.write("Sink frequency by layer:\n")
        for layer_idx, freq in enumerate(sink_by_layer):
            f.write(f"Layer {layer_idx}: {freq:.2f}%\n")
    
    # Create a combined plot with all 4 visualizations
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Average sink scores heatmap
    im1 = axs[0, 0].imshow(avg_sink_scores, cmap="viridis", aspect="auto")
    axs[0, 0].set_xlabel("Head")
    axs[0, 0].set_ylabel("Layer")
    axs[0, 0].set_title("Average Attention Sink Scores")
    fig.colorbar(im1, ax=axs[0, 0])
    
    # Sink mask heatmap
    im2 = axs[0, 1].imshow(sink_mask, cmap="binary", aspect="auto")
    axs[0, 1].set_xlabel("Head")
    axs[0, 1].set_ylabel("Layer")
    axs[0, 1].set_title("Attention Sink Mask")
    fig.colorbar(im2, ax=axs[0, 1])
    
    # Sink frequency by layer
    bars3 = axs[1, 0].bar(range(len(sink_by_layer)), sink_by_layer)
    for bar in bars3:
        height = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    axs[1, 0].set_xlabel("Layer")
    axs[1, 0].set_ylabel("Percentage of Sink Heads")
    axs[1, 0].set_title("Frequency of Attention Sinks by Layer")
    
    # Average sink score by layer
    bars4 = axs[1, 1].bar(range(len(avg_score_by_layer)), avg_score_by_layer)
    for bar in bars4:
        height = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    axs[1, 1].set_xlabel("Layer")
    axs[1, 1].set_ylabel("Average Sink Score")
    axs[1, 1].set_title("Average Attention Sink Score by Layer")
    
    plt.suptitle(f"Attention Sink Analysis for Simulated Llama-3.1-8B (threshold={threshold})", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f"{output_dir}/combined_analysis.png", dpi=300)
    plt.close()

def main(args):
    # Simulate attention patterns
    all_results = simulate_attention_patterns(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_queries=args.num_queries,
        sequence_lengths=args.sequence_lengths,
        output_dir=args.output_dir
    )
    
    # Calculate average results
    avg_sink_scores = average_results(all_results)
    
    # Visualize the results
    visualize_sink_frequency(avg_sink_scores, args.threshold, args.output_dir)
    
    # Save individual query results
    with open(f"{args.output_dir}/query_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSimulation complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate attention sink patterns for Llama-3.1-8B")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of transformer layers in the model")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of attention heads per layer")
    parser.add_argument("--num-queries", type=int, default=5,
                        help="Number of different queries to simulate")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[5, 10, 15, 20, 25],
                        help="Sequence lengths for each query")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Threshold to classify as attention sink")
    parser.add_argument("--output-dir", type=str, default="simulated_8b_results",
                        help="Directory to save simulation results")
    
    args = parser.parse_args()
    main(args)