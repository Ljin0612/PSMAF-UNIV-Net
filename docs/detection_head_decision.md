# Detection head decision

The original UNIV paper reports downstream detection with Mask R-CNN, but the
uploaded source snapshot may not include a complete, ready-to-run Mask R-CNN
detection implementation. Consequently, the first detection direction should be a
Mask R-CNN- or Faster R-CNN-style head only if the project implements or integrates
a compatible detection pipeline.

Once that pipeline exists, Mask R-CNN is the preferred primary head and Faster R-CNN
is the box-only ablation. Both should consume the same named PSMAF-UNIV feature
pyramid. This pairing supports a close, controlled comparison and separates backbone
quality from real-time detector design. YOLO remains a later auxiliary real-time
extension, not the first mainline proof.
