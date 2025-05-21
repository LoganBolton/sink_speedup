# Attention Sink Speedup: Code Explanation

This document provides a detailed explanation of the codebase structure and methodology used in the attention sink speedup project.

## Project Overview

The project investigates whether we can optimize LLM inference by identifying and reusing attention sink patterns. The core hypothesis is that if the first layer of an attention head is an attention sink (with most attention directed to the first token), subsequent layers in that head are likely to also be attention sinks.

## Codebase Structure

The project consists of several Python scripts, each with a specific purpose:

```
sink_speedup/
├── analyze_attention_sinks.py     # Analyzes real model attention patterns
├── analyze_multiple_queries.py    # Tests multiple prompts and averages results
├── simulate_8b_model.py           # Simulates larger model behavior
├── visualize_sink_patterns.py     # Creates enhanced visualizations
├── optimized_attention.py         # Proof-of-concept optimization
├── compare_models.py              # Compares results across different settings
└── findings.md                    # Documents key findings
```

## 1. Attention Sink Analysis (`analyze_attention_sinks.py`)

This script forms the core of the analysis methodology.

### Key Components

1. **Model Loading**
   ```python
   def load_model_and_tokenizer(model_name: str):
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForCausalLM.from_pretrained(
           model_name, 
           torch_dtype=torch.float32,
           device_map=None,
           attn_implementation="eager"  # Ensure we can access attention patterns
       )
       model.to(device)
       return model, tokenizer
   ```
   - Loads the specified LLM model and tokenizer
   - Uses `attn_implementation="eager"` to ensure attention patterns are accessible
   - Sets appropriate data types and device mapping

2. **Attention Pattern Analysis**
   ```python
   def get_attention_patterns(model, input_ids, threshold: float = 0.9):
       # Initialize results storage
       results = { ... }
       
       # Forward pass with output_attentions=True to get attention patterns
       with torch.no_grad():
           outputs = model(input_ids, output_attentions=True)
       attentions = outputs.attentions
       
       # Process each layer's attention pattern
       for layer_idx, layer_attention in enumerate(attentions):
           # Process each head in the layer
           for head_idx in range(num_heads):
               # Calculate proportion of attention on first token
               first_token_attention = attn_matrix[:, 0].sum() / attn_matrix.sum()
               
               # Determine if this head is an attention sink
               is_sink = first_token_attention >= threshold
   ```
   - Runs the model with `output_attentions=True` to capture attention weights
   - For each layer and head, computes what proportion of attention is on the first token
   - Classifies heads as "attention sinks" if they exceed the threshold (default 0.9)

3. **Correlation Analysis**
   ```python
   def analyze_sink_correlation(results: Dict) -> Dict:
       # For each head, check if first layer predicts subsequent layers
       for head_idx in range(num_heads):
           # Get sink pattern across all layers
           head_pattern = [results["layer_head_is_sink"][layer_idx][head_idx] 
                           for layer_idx in range(num_layers)]
           
           # If first layer is sink, check correlation with rest
           if head_pattern[0]:
               rest_are_sinks = sum(head_pattern[1:]) / (num_layers - 1)
   ```
   - For each attention head, checks if the first layer being an attention sink predicts the same behavior in subsequent layers
   - Calculates the percentage of subsequent layers that match the first layer's sink behavior

4. **Visualization Generation**
   ```python
   def visualize_attention_patterns(results: Dict, save_dir: str, prompt: str):
       # Create heatmap of attention sink scores
       plt.imshow(layer_scores, cmap="viridis", aspect="auto")
       
       # Visualize example attention maps
       for layer_idx in range(num_layers):
           for head_idx in range(num_heads):
               plt.imshow(attn_map, cmap="viridis")
   ```
   - Creates heatmaps showing attention sink scores across layers and heads
   - Generates example visualizations of specific attention maps

## 2. Multiple Query Analysis (`analyze_multiple_queries.py`)

This script extends the analysis to multiple prompts and aggregates the results.

### Key Components

1. **Query Processing**
   ```python
   def process_multiple_queries(model, tokenizer, prompts, threshold=0.9, output_dir="multiple_query_results"):
       # Process each prompt
       for i, prompt in enumerate(prompts):
           input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
           results = get_attention_patterns(model, input_ids, threshold)
           
           # Store results for this query
           query_result = {
               "prompt": prompt,
               "seq_length": input_ids.size(1),
               "sink_percentage": results["total_sink_heads"] / results["total_heads"] * 100,
               "layer_head_sink_score": results["layer_head_sink_score"]
           }
           all_results.append(query_result)
   ```
   - Processes a list of different prompts through the model
   - Collects attention sink data for each prompt
   - Stores results including sequence length and sink percentages

2. **Result Aggregation**
   ```python
   def average_results(all_results):
       # Initialize average scores
       avg_sink_scores = np.zeros((num_layers, num_heads))
       
       # Add up all scores
       for result in all_results:
           for layer_idx in range(num_layers):
               for head_idx in range(num_heads):
                   avg_sink_scores[layer_idx, head_idx] += result["layer_head_sink_score"][layer_idx][head_idx]
       
       # Divide by number of queries
       avg_sink_scores /= len(all_results)
   ```
   - Aggregates sink scores across all queries
   - Computes the average sink score for each layer-head combination

## 3. Larger Model Simulation (`simulate_8b_model.py`)

Since directly analyzing a large 8B model is challenging due to memory constraints, this script simulates attention patterns based on observed behaviors.

### Key Components

1. **Pattern Simulation**
   ```python
   def simulate_attention_patterns(
       num_layers=32,  # Llama-3.1-8B has 32 layers
       num_heads=32,   # Llama-3.1-8B has 32 attention heads per layer
       num_queries=5,
       sequence_lengths=[5, 10, 15, 20, 25],
       output_dir="simulated_8b_results"
   ):
       # Create base sink probability by layer (decreasing as layer increases)
       layer_sink_base_probs = np.linspace(0.9, 0.2, num_layers)
       
       # Create head sink consistency (some heads more likely to be sinks)
       head_sink_tendency = np.random.beta(2, 3, size=num_heads)
       
       # Process each prompt
       for i, (prompt, seq_len) in enumerate(zip(prompts, sequence_lengths)):
           # Sequence length effect - longer sequences decrease sink probability
           seq_length_factor = max(0.2, 1.0 - 0.03 * seq_len)
           
           # For each layer and head
           for layer_idx in range(num_layers):
               layer_prob = layer_sink_base_probs[layer_idx] * seq_length_factor
               
               for head_idx in range(num_heads):
                   # Calculate sink probability based on layer, head, and sequence length
                   head_tendency = head_sink_tendency[head_idx]
                   sink_prob = layer_prob * head_tendency
   ```
   - Simulates attention patterns for a large model based on empirical observations
   - Incorporates several key observed behaviors:
     - Earlier layers have more attention sinks
     - Shorter sequences have more attention sinks
     - Some heads consistently behave as attention sinks
     - Sink behavior decays as sequence length increases

## 4. Advanced Visualizations (`visualize_sink_patterns.py`)

This script creates enhanced visualizations to better understand attention sink patterns.

### Key Components

1. **Layer-Wise Analysis**
   ```python
   # Calculate average sink score by layer
   avg_score_by_layer = avg_sink_scores.mean(axis=1)
   
   # Create bars
   bars = plt.bar(range(num_layers), avg_score_by_layer, color='skyblue')
   
   # Add the threshold line
   plt.axhline(y=threshold, color='r', linestyle='--')
   ```
   - Shows the average attention sink score for each layer
   - Visualizes which layers tend to have more attention sinks

2. **Heatmap with Threshold Contour**
   ```python
   # Create a heatmap with improved visibility
   heatmap = plt.imshow(avg_sink_scores, cmap='viridis', aspect='auto')
   
   # Add threshold contour
   contour_levels = [threshold]
   contour = plt.contour(np.arange(num_heads) + 0.5, np.arange(num_layers) + 0.5, 
                         avg_sink_scores, contour_levels, colors='red')
   ```
   - Creates a detailed heatmap of attention patterns
   - Adds contour lines to clearly mark the threshold boundary

3. **Sequence Length Analysis**
   ```python
   # If we have query_results, visualize the trend of sink percentage by sequence length
   if query_results:
       seq_lengths = [result['seq_length'] for result in query_results]
       sink_percentages = [result['sink_percentage'] for result in query_results]
       
       plt.plot(seq_lengths, sink_percentages, 'o-', linewidth=2, markersize=10)
   ```
   - Plots the relationship between sequence length and attention sink percentages
   - Helps visualize how attention patterns change with input length

## 5. Optimization Implementation (`optimized_attention.py`)

This script provides a proof-of-concept implementation of the optimization strategy.

### Key Components

1. **Optimized Attention Mechanism**
   ```python
   class OptimizedSelfAttention(torch.nn.Module):
       def forward(self, hidden_states, attention_mask=None, layer_idx=None, use_optimization=True):
           # First layer or optimization disabled: compute normally
           if layer_idx == 0 or not use_optimization:
               attn_output, attn_weights = self._compute_attention(query, key, value, attention_mask)
               
               # If this is the first layer, save patterns for potential reuse
               if layer_idx == 0:
                   self.is_sink_head = self._is_attention_sink(attn_weights)
                   self.first_layer_patterns = attn_weights
           else:
               # For subsequent layers, check which heads are attention sinks
               for head_idx in range(self.num_heads):
                   head_is_sink = self.is_sink_head[:, head_idx].all()
                   
                   if head_is_sink and self.first_layer_patterns is not None:
                       # Reuse the attention pattern from first layer
                       head_attn_weights = self.first_layer_patterns[:, head_idx:head_idx+1]
                       head_attn_output = torch.matmul(head_attn_weights, value[:, head_idx:head_idx+1])
                   else:
                       # Compute normally for this head
                       head_attn_output, head_attn_weights = self._compute_attention(...)
   ```
   - Implements a customized self-attention module that:
     - Computes attention normally for the first layer
     - Identifies which heads are attention sinks
     - For subsequent layers, reuses attention patterns for sink heads instead of recomputing
     - Computes attention normally for non-sink heads

2. **Sink Detection Logic**
   ```python
   def _is_attention_sink(self, attn_weights):
       # Sum attention on first token for each head
       first_token_attention = attn_weights[:, :, :, 0].sum(dim=2)
       # Sum total attention for each head
       total_attention = attn_weights.sum(dim=-1).sum(dim=-1)
       
       # Calculate proportion of attention on first token
       sink_scores = first_token_attention / total_attention
       
       # Check if above threshold
       return sink_scores >= self.sink_threshold
   ```
   - Determines which heads exhibit attention sink behavior
   - Uses the proportion of attention on first token as the key metric

## 6. Model Comparison (`compare_models.py`)

This script compares results across different model sizes or sequence lengths.

### Key Components

1. **Result Loading and Parsing**
   ```python
   def load_results(results_dir_1, results_dir_2):
       def parse_summary(filepath):
           data = {}
           with open(filepath, 'r') as f:
               content = f.read()
               
               # Extract prompt, total heads, sink heads, percentage, and correlation
               # ...
               
           return data
       
       results1 = parse_summary(os.path.join(results_dir_1, 'summary.txt'))
       results2 = parse_summary(os.path.join(results_dir_2, 'summary.txt'))
   ```
   - Loads results from different model runs
   - Parses summary statistics from each run

2. **Comparative Visualization**
   ```python
   def visualize_comparison(results1, results2, output_dir='comparison_results'):
       # Create bar chart comparing sink percentages and correlations
       labels = [f"Seq Len: {results1['seq_length']}", f"Seq Len: {results2['seq_length']}"]
       
       # Plot sink percentages
       sink_percentages = [results1['percentage'], results2['percentage']]
       ax[0].bar(labels, sink_percentages, color=['blue', 'orange'])
       
       # Plot correlation percentages
       correlation_percentages = [results1['correlation'], results2['correlation']]
       ax[1].bar(labels, correlation_percentages, color=['blue', 'orange'])
   ```
   - Creates side-by-side comparisons of key metrics
   - Helps visualize how different factors affect attention patterns

## Implementation Flow

The overall workflow of the codebase follows these steps:

1. **Analysis**: Examine attention patterns in a real model (`analyze_attention_sinks.py`)
2. **Multi-Query Testing**: Test with multiple prompts to confirm patterns (`analyze_multiple_queries.py`)
3. **Model Scale Simulation**: Simulate behavior for larger models (`simulate_8b_model.py`)
4. **Visualization**: Create detailed visualizations of findings (`visualize_sink_patterns.py`)
5. **Optimization**: Implement a proof-of-concept optimization (`optimized_attention.py`)
6. **Comparison**: Compare results across different settings (`compare_models.py`)

## Optimization Strategy

The optimization is based on the following principle:

1. During the forward pass of the first layer, identify which heads are attention sinks
2. For subsequent layers, if a head was an attention sink in the first layer:
   - Skip the expensive attention computation (which is O(n²) with sequence length)
   - Instead, reuse the attention pattern from the first layer
   - Compute the output by applying this reused pattern to the current values
3. For non-sink heads, compute attention normally

This approach is dynamically adjusted based on:
- Sequence length (disabled for longer sequences)
- Layer-specific behaviors (some layers benefit more than others)
- Individual head behaviors (only optimize consistently sink-behaving heads)

## Potential Speedup Impact

The optimization primarily converts O(n²) operations to O(1) for qualifying heads:

- For a model where 30% of heads are attention sinks, this could reduce attention computation by approximately 30%
- The actual speedup would be less than 30% of overall inference time because:
  - Attention is only part of the overall computation
  - The optimization adds overhead to check which heads qualify
  - Only certain layers and sequence lengths benefit significantly

For practical implementation, the optimization should be selectively applied to maximize benefit while minimizing potential accuracy impacts.