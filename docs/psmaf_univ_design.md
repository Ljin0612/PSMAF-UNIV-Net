# PSMAF-UNIV design

The new package is deliberately head-neutral. `UNIVEncoderWrapper` normalizes encoder
outputs, `MultiscaleTaskAdapter` projects each level to a shared channel width,
`PseudoSemanticGuidance` estimates spatial confidence, and `PSMAFFusion` uses that
confidence when injecting coarse context into finer levels. `PSMAFUNIVModel` composes
those pieces and returns a fine-to-coarse tuple. Detection and segmentation adapters
only translate this tuple into the interface expected by their task framework.

This is a Stage-3 structural baseline, not a claim that the proposed training recipe
or final architecture has been experimentally validated.
