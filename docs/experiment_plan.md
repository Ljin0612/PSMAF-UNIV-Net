# Experiment plan

1. Verify original UNIV checkpoint loading and reproduce its segmentation baseline.
2. Freeze the encoder and train task adapters to establish an interface baseline.
3. Compare full fine-tuning, LoRA, multiscale adaptation, semantic guidance, and
   adaptive fusion under matched schedules and seeds.
4. Evaluate Mask R-CNN and Faster R-CNN on M3FD, then UPerNet on the selected
   segmentation benchmark.
5. Report accuracy, parameter count, throughput, memory, per-class results, and
   mean/standard deviation across seeds. Store machine-readable outputs outside Git
   and aggregate them with `tools/collect_results.py` once its schema is finalized.
