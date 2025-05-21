#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from typing import Dict, List, Tuple

# Set up device - force CPU usage to avoid MPS/CUDA issues
device = torch.device("cpu")
print(f"Using device: {device}")

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # Don't use device map
        attn_implementation="eager"  # Ensure we can access attention patterns
    )
    model.to(device)  # Explicitly move to CPU
    return model, tokenizer

def get_attention_patterns(model, input_ids, threshold: float = 0.9):
    """
    Compute attention patterns for each head/layer and determine if they're attention sinks.
    
    Args:
        model: The model to analyze
        input_ids: Tokenized input
        threshold: Threshold to classify as attention sink (proportion of attention on first token)
        
    Returns:
        Dictionary with attention sink analysis
    """
    # Create empty dict to store results
    results = {
        "layer_head_is_sink": [],  # Boolean mask of which heads are sinks
        "layer_head_sink_score": [],  # Proportion of attention on first token
        "total_heads": 0,
        "total_sink_heads": 0,
        "attention_maps": []  # Store actual attention maps for visualization
    }
    
    # Forward pass with output_attentions=True to get attention patterns
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Get all attention tensors (layers, batch, heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # Process each layer's attention pattern
    for layer_idx, layer_attention in enumerate(attentions):
        # Layer attention shape: (batch, num_heads, seq_len, seq_len)
        layer_is_sink = []
        layer_sink_score = []
        layer_attention_maps = []
        
        # Process each head
        num_heads = layer_attention.size(1)
        for head_idx in range(num_heads):
            # Get attention matrix for this head (seq_len, seq_len)
            attn_matrix = layer_attention[0, head_idx].cpu().numpy()
            layer_attention_maps.append(attn_matrix)
            
            # Calculate proportion of attention on first token (column sum normalized)
            first_token_attention = attn_matrix[:, 0].sum() / attn_matrix.sum()
            layer_sink_score.append(float(first_token_attention))
            
            # Determine if this head is an attention sink
            is_sink = first_token_attention >= threshold
            layer_is_sink.append(bool(is_sink))
            
            if is_sink:
                results["total_sink_heads"] += 1
            results["total_heads"] += 1
        
        results["layer_head_is_sink"].append(layer_is_sink)
        results["layer_head_sink_score"].append(layer_sink_score)
        results["attention_maps"].append(layer_attention_maps)
    
    return results

def process_multiple_queries(model, tokenizer, prompts, threshold=0.9, output_dir="multiple_query_results"):
    """Process multiple queries and collect attention sink data"""
    os.makedirs(output_dir, exist_ok=True)

    # Store all query results
    all_results = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: '{prompt}'")

        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Input sequence length: {input_ids.size(1)}")

        # Get attention patterns
        results = get_attention_patterns(model, input_ids, threshold)

        # Store results
        query_result = {
            "prompt": prompt,
            "seq_length": input_ids.size(1),
            "sink_percentage": results["total_sink_heads"] / results["total_heads"] * 100,
            "layer_head_sink_score": results["layer_head_sink_score"],
            "attention_maps": results["attention_maps"]  # Include attention maps for visualization
        }
        all_results.append(query_result)

        print(f"Sink heads: {results['total_sink_heads']}/{results['total_heads']} "
              f"({query_result['sink_percentage']:.2f}%)")

    return all_results

def average_results(all_results):
    """Calculate average sink scores across all queries"""
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

def visualize_sink_frequency(avg_sink_scores, all_results, threshold=0.9, output_dir="multiple_query_results"):
    """Create visualizations of sink frequencies"""
    os.makedirs(output_dir, exist_ok=True)

    # Save the numerical data
    np.save(f"{output_dir}/avg_sink_scores.npy", avg_sink_scores)

    # Create a mask of which heads exceed the threshold on average
    sink_mask = avg_sink_scores >= threshold

    # Save a summary report
    sink_percentage = sink_mask.mean() * 100

    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"Average Sink Analysis Across Multiple Queries\n\n")
        f.write(f"Total percentage of attention sink heads: {sink_percentage:.2f}%\n\n")
        f.write("Sink frequency by layer:\n")
        for layer_idx, freq in enumerate(sink_by_layer := sink_mask.mean(axis=1) * 100):
            f.write(f"Layer {layer_idx}: {freq:.2f}%\n")

    # Visualize attention patterns for every head in every other layer
    if len(all_results) > 0:
        # Get the first result that has attention maps
        result = all_results[0]

        # Get attention maps from the results
        attention_maps = result.get("attention_maps", [])

        if attention_maps:
            # Get dimensions
            num_layers = len(attention_maps)
            num_heads = len(attention_maps[0]) if num_layers > 0 else 0

            if num_layers > 0 and num_heads > 0:
                # Select every other layer
                selected_layers = list(range(0, num_layers, 2))

                # Calculate grid size
                fig_width = max(20, num_heads * 2)  # Scale width based on number of heads
                fig_height = max(15, len(selected_layers) * 2)  # Scale height based on number of layers

                # Create a grid of subplots
                fig, axs = plt.subplots(len(selected_layers), num_heads,
                                        figsize=(fig_width, fig_height),
                                        squeeze=False)

                # Plot each head's attention pattern for selected layers
                for i, layer_idx in enumerate(selected_layers):
                    for head_idx in range(num_heads):
                        attn_map = attention_maps[layer_idx][head_idx]
                        ax = axs[i, head_idx]
                        im = ax.imshow(attn_map, cmap="viridis", aspect="auto")

                        # Only add row/column labels on the edges
                        if head_idx == 0:
                            ax.set_ylabel(f"Layer {layer_idx}")
                        else:
                            ax.set_yticks([])

                        if i == 0:
                            ax.set_title(f"Head {head_idx}")
                        else:
                            ax.set_xticks([])

                # Add a colorbar on the right side of the figure
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax, label="Attention Weight")

                # Set a main title
                fig.suptitle(f"Attention Patterns for Each Head (Every Other Layer)", fontsize=16)

                # Adjust layout and save
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect to account for title and colorbar
                plt.savefig(f"{output_dir}/all_heads_attention_patterns.png", dpi=150, bbox_inches='tight')
                plt.close()

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Define prompts
    prompts = [
        "Heyyyyy what's going on Llama? How are you doing today?",
        # "The quick brown fox jumps over the lazy dog",
        # "I went to the store to buy some groceries",
        # "Artificial intelligence is transforming the way we live and work",
        # "The rain in Spain falls mainly on the plain"
    ]
    
    # Process all queries
    all_results = process_multiple_queries(model, tokenizer, prompts, args.threshold, args.output_dir)
    
    # Calculate average results
    avg_sink_scores = average_results(all_results)
    
    # Visualize the results
    visualize_sink_frequency(avg_sink_scores, all_results, args.threshold, args.output_dir)
    
    # Save individual query results
    with open(f"{args.output_dir}/query_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in all_results:
            serializable_result = {
                "prompt": result["prompt"],
                "seq_length": result["seq_length"],
                "sink_percentage": result["sink_percentage"],
                "layer_head_sink_score": [[float(score) for score in layer] 
                                         for layer in result["layer_head_sink_score"]]
            }
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze attention sinks across multiple queries")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name or path")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Threshold to classify as attention sink (proportion of attention on first token)")
    parser.add_argument("--output-dir", type=str, default="multiple_query_results",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    main(args)