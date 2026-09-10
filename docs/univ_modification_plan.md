# UNIV modification plan

1. Keep `UNIV-main/` immutable and traceable.
2. Instantiate the original ConvMAE implementation externally and pass it to
   `PSMAFUNIVModel`; do not copy its internals into the new package.
3. Validate which intermediate stages are stable public outputs before adding a
   minimal hook/subclass in new code.
4. Load original student checkpoints through `checkpoint_loader.py` and record all
   missing/unexpected keys.
5. Apply optional LoRA only through `lora_adapter.py`.

Any unavoidable upstream fix should be isolated, documented, and covered by a test.
