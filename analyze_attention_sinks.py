#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import os

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

def analyze_sink_correlation(results: Dict) -> Dict:
    """
    Analyze: If first layer of a head is an attention sink, 
    how likely is it that the rest of the layers are also sinks?
    
    Args:
        results: Dictionary with attention sink analysis
        
    Returns:
        Dictionary with correlation analysis
    """
    num_layers = len(results["layer_head_is_sink"])
    num_heads = len(results["layer_head_is_sink"][0])
    
    correlation = {
        "first_layer_sink_predicts_rest": [],
        "sink_patterns_by_head": [],
        "overall_correlation": 0.0
    }
    
    for head_idx in range(num_heads):
        # Get sink pattern for this head across all layers
        head_pattern = [results["layer_head_is_sink"][layer_idx][head_idx] 
                        for layer_idx in range(num_layers)]
        correlation["sink_patterns_by_head"].append(head_pattern)
        
        # If first layer is sink, check correlation with rest
        if head_pattern[0]:
            rest_are_sinks = sum(head_pattern[1:]) / (num_layers - 1)
            correlation["first_layer_sink_predicts_rest"].append(rest_are_sinks)
    
    # Calculate overall correlation: if first layer is sink, what percentage of other layers are sinks
    if correlation["first_layer_sink_predicts_rest"]:
        correlation["overall_correlation"] = sum(correlation["first_layer_sink_predicts_rest"]) / len(correlation["first_layer_sink_predicts_rest"])
    
    return correlation

def visualize_attention_patterns(results: Dict, save_dir: str, prompt: str):
    """
    Visualize attention patterns and save the plots.
    
    Args:
        results: Dictionary with attention sink analysis
        save_dir: Directory to save visualizations
        prompt: The input prompt used
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a summary plot of sink scores
    plt.figure(figsize=(12, 8))
    layer_scores = np.array(results["layer_head_sink_score"])
    
    # Plot heatmap of attention sink scores
    plt.imshow(layer_scores, cmap="viridis", aspect="auto")
    plt.colorbar(label="Proportion of attention on first token")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention Sink Scores (threshold={args.threshold})")
    
    # Add grid lines
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sink_scores_heatmap.png")
    
    # Save a few example attention maps for the first token
    num_layers = min(3, len(results["attention_maps"]))
    num_heads = min(3, len(results["attention_maps"][0]))
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            plt.figure(figsize=(10, 8))
            attn_map = results["attention_maps"][layer_idx][head_idx]
            
            plt.imshow(attn_map, cmap="viridis")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Layer {layer_idx}, Head {head_idx} Attention Map")
            plt.xlabel("Token Position (column)")
            plt.ylabel("Token Position (row)")
            
            # Add text for sink score
            sink_score = results["layer_head_sink_score"][layer_idx][head_idx]
            is_sink = results["layer_head_is_sink"][layer_idx][head_idx]
            plt.text(0.5, 0.95, f"Sink Score: {sink_score:.3f} ({'Yes' if is_sink else 'No'})",
                    transform=plt.gca().transAxes, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/attention_map_layer{layer_idx}_head{head_idx}.png")
            plt.close()
    
    # Write summary stats to file
    with open(f"{save_dir}/summary.txt", "w") as f:
        f.write(f"Input prompt: {prompt}\n\n")
        f.write(f"Total attention heads: {results['total_heads']}\n")
        f.write(f"Total attention sink heads: {results['total_sink_heads']}\n")
        f.write(f"Percentage of attention sink heads: {results['total_sink_heads'] / results['total_heads'] * 100:.2f}%\n\n")
        
        correlation = analyze_sink_correlation(results)
        f.write("Correlation Analysis:\n")
        f.write(f"If first layer is attention sink, {correlation['overall_correlation'] * 100:.2f}% "
                f"of subsequent layers in that head are also attention sinks.\n")

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Tokenize input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"Input sequence length: {input_ids.size(1)}")
    
    # Compute attention patterns and analyze attention sinks
    results = get_attention_patterns(model, input_ids, args.threshold)
    
    # Analyze correlation between first layer and subsequent layers
    correlation = analyze_sink_correlation(results)
    
    # Print summary statistics
    print(f"\nTotal attention heads: {results['total_heads']}")
    print(f"Total attention sink heads: {results['total_sink_heads']}")
    print(f"Percentage of attention sink heads: {results['total_sink_heads'] / results['total_heads'] * 100:.2f}%")
    
    print("\nCorrelation Analysis:")
    print(f"If first layer is attention sink, {correlation['overall_correlation'] * 100:.2f}% "
          f"of subsequent layers in that head are also attention sinks.")
    
    # Visualize attention patterns
    visualize_attention_patterns(results, args.output_dir, args.prompt)
    
    print(f"\nAnalysis complete. Visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze attention sinks in LLM models")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name or path")
    parser.add_argument("--prompt", type=str, 
                        default="The quick brown fox jumps over the lazy dog.",
                        help="Input prompt for analysis")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Threshold to classify as attention sink (proportion of attention on first token)")
    parser.add_argument("--output-dir", type=str, default="attention_analysis",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    main(args)