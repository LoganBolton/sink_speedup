# Attention Sink Optimization for LLM Inference

Experimenting with optimizing LLM inference by identifying and reusing attention patterns in attention sink heads.

## Overview

There tends to be two types of attention maps for attention heads:

1) **Attention Sink**
    - All the attention is directed towards the first token in the sequence
    - Easier to compute
2) **Self Attention**
    - All the attention for this head is focused on the current token in the sequence
    - Slightly harder to compute. Sometimes there's self attention heads where it bleeds over to nearby tokens

## Hypothesis

When you are doing inference for an LLM, if the first layer of a head exhibits attention sink behavior, subsequent layers in that head are likely to also be attention sinks. Therefore, you don't need to compute the attention values for the rest of these layers. You can just assume they will look like the first layer, and copy over the first layer's attention pattern values. This **should** cause a speed up in inference speed because you can compute attention values in O(1) instead of O(N^2).


# Notes
So it.... doesn't really work for small LLMs? The consitent sink heads thing I noticed when I was working on my graph project was probably just due to the types of specialized prompts I was running.

## TODO
- Try with a bigger model
    - 32B or 70B?
- Try with more specialized prompts?
    - Maybe coding or math problems or simple QA have more distinct patterns?

# Claude's Notes

## Implementation and Results

We've implemented:

1. **Analysis Scripts**:
   - `analyze_attention_sinks.py`: Analyzes attention patterns in Llama models
   - `compare_models.py`: Compares results across different sequence lengths
   - `optimized_attention.py`: Proof-of-concept implementation of the optimization

2. **Key Findings**:
   - For short sequences (1-2 tokens):
     - ~71% of heads exhibit attention sink behavior
     - ~74% correlation between first layer and subsequent layers
   - For longer sequences (20+ tokens):
     - Only ~23% of heads exhibit attention sink behavior
     - Only ~27% correlation between first layer and subsequent layers
   - See `findings.md` for detailed analysis

3. **Optimization Potential**:
   - Significant speedup potential for short sequences
   - Limited effectiveness for longer sequences
   - Best applied during early generation steps or with a hybrid approach

## Running the Analysis

1. **Setup Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch transformers accelerate matplotlib
   ```

2. **Analyze Attention Patterns**:
   ```bash
   python analyze_attention_sinks.py --prompt "Your test prompt" --output-dir "results"
   ```

3. **Compare Results**:
   ```bash
   python compare_models.py --dir1 "results1" --dir2 "results2" --output "comparison"
   ```

4. **Test Optimization**:
   ```bash
   python optimized_attention.py
   ```

## Next Steps

1. Implement the optimization in a full LLM inference pipeline
2. Benchmark performance gains on common generation tasks
3. Explore dynamic thresholds based on sequence length
4. Investigate more sophisticated pattern identification for longer sequences

## References

- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

