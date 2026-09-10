# Detection head decision

Mask R-CNN is the primary detection head, with Faster R-CNN as the box-only ablation.
Both consume the same named PSMAF-UNIV feature pyramid. This pairing supports a close,
controlled comparison and separates backbone quality from real-time detector design.
YOLOv8 remains explicitly downstream and deferred.
