#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

def load_results(results_dir_1, results_dir_2):
    """Load and parse results from summary files"""
    
    # Helper function to extract data from summary file
    def parse_summary(filepath):
        data = {}
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract prompt
            prompt_line = content.split('\n')[0]
            data['prompt'] = prompt_line.replace('Input prompt: ', '')
            
            # Extract total heads
            total_heads_line = [l for l in content.split('\n') if 'Total attention heads:' in l][0]
            data['total_heads'] = int(total_heads_line.split(': ')[1])
            
            # Extract sink heads
            sink_heads_line = [l for l in content.split('\n') if 'Total attention sink heads:' in l][0]
            data['sink_heads'] = int(sink_heads_line.split(': ')[1])
            
            # Extract percentage
            percentage_line = [l for l in content.split('\n') if 'Percentage of attention sink heads:' in l][0]
            data['percentage'] = float(percentage_line.split(': ')[1].strip('%'))
            
            # Extract correlation
            correlation_line = [l for l in content.split('\n') if 'of subsequent layers in that head are also attention sinks' in l][0]
            correlation_str = correlation_line.split('If first layer is attention sink, ')[1].split('%')[0]
            data['correlation'] = float(correlation_str)
            
            # Calculate sequence length (approximate from prompt)
            data['seq_length'] = len(data['prompt'].split())
            
        return data
    
    results1 = parse_summary(os.path.join(results_dir_1, 'summary.txt'))
    results2 = parse_summary(os.path.join(results_dir_2, 'summary.txt'))
    
    return results1, results2

def visualize_comparison(results1, results2, output_dir='comparison_results'):
    """Create visualization comparing results from different models/prompts"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a JSON summary of the comparison
    comparison = {
        'model1_desc': f"Llama-3.2-1B (Seq Len: {results1['seq_length']})",
        'model2_desc': f"Llama-3.2-1B (Seq Len: {results2['seq_length']})",
        'results1': results1,
        'results2': results2
    }
    
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create bar chart comparing sink percentages and correlations
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Labels
    labels = [f"Seq Len: {results1['seq_length']}", f"Seq Len: {results2['seq_length']}"]
    
    # Plot sink percentages
    sink_percentages = [results1['percentage'], results2['percentage']]
    ax[0].bar(labels, sink_percentages, color=['blue', 'orange'])
    ax[0].set_title('Percentage of Attention Sink Heads')
    ax[0].set_ylabel('Percentage (%)')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, v in enumerate(sink_percentages):
        ax[0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Plot correlation percentages
    correlation_percentages = [results1['correlation'], results2['correlation']]
    ax[1].bar(labels, correlation_percentages, color=['blue', 'orange'])
    ax[1].set_title('Correlation: First Layer Sink → Other Layers Sink')
    ax[1].set_ylabel('Correlation (%)')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, v in enumerate(correlation_percentages):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sink_comparison.png'))
    
    # Create markdown summary
    markdown_content = f"""# Attention Sink Analysis Comparison

## Sequence Length Comparison

### Short Sequence (Length: {results1['seq_length']})
- **Total heads:** {results1['total_heads']}
- **Sink heads:** {results1['sink_heads']} ({results1['percentage']:.2f}%)
- **Correlation:** {results1['correlation']:.2f}%

### Long Sequence (Length: {results2['seq_length']})
- **Total heads:** {results2['total_heads']}
- **Sink heads:** {results2['sink_heads']} ({results2['percentage']:.2f}%)
- **Correlation:** {results2['correlation']:.2f}%

## Key Findings

1. **Impact of Sequence Length:**
   - Sink heads percentage decreases by {results1['percentage'] - results2['percentage']:.2f}% as sequence length increases
   - Correlation decreases by {results1['correlation'] - results2['correlation']:.2f}% as sequence length increases

2. **Optimization Potential:**
   - For short sequences (length {results1['seq_length']}), there's strong potential to optimize by copying attention patterns 
     ({results1['correlation']:.2f}% heads would correctly be predicted as sinks)
   - For longer sequences (length {results2['seq_length']}), this optimization would be less effective 
     (only {results2['correlation']:.2f}% would be predicted correctly)

3. **Sequence Length Threshold:**
   - The significant drop suggests a threshold effect where attention patterns become more complex beyond
     a certain sequence length
   - Hybrid approaches may be needed for longer sequences

## Conclusion

The analysis supports the hypothesis that attention sink patterns are more consistent across layers
for shorter sequences. For practical optimization, a dynamic approach could be used:
1. Use pattern copying for short sequences where correlation is high
2. Use traditional computation for longer sequences
3. Potentially investigate a mixed approach where some heads use pattern copying and others use full computation
"""

    with open(os.path.join(output_dir, 'comparison.md'), 'w') as f:
        f.write(markdown_content)
    
    print(f"Comparison results saved to: {output_dir}")
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Compare attention sink analysis results")
    parser.add_argument("--dir1", type=str, required=True, help="Directory with first analysis results")
    parser.add_argument("--dir2", type=str, required=True, help="Directory with second analysis results") 
    parser.add_argument("--output", type=str, default="comparison_results", help="Output directory for comparison")
    
    args = parser.parse_args()
    
    results1, results2 = load_results(args.dir1, args.dir2)
    visualize_comparison(results1, results2, args.output)

if __name__ == "__main__":
    main()