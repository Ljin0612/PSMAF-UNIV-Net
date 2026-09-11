# Stage 3: UNIV + MTA single-stream detection baseline

## Scope and purpose

Stage 3 establishes the smallest downstream detection boundary before expensive
training. It uses **M3FD-IR first**, one stream only, an original UNIV student
or teacher backbone, and `MultiScaleTaskAdapter`. The adapter exposes P3, P4,
and P5 to a placeholder compatibility head. This stage verifies data loading,
checkpoint loading, feature extraction, pyramid construction, and detector input
compatibility; it does **not** measure mAP or claim a trained baseline.

RGB–IR paired fusion, PSMAF, PSG/MSAF ablations, YOLO integration, and automatic
long-running training are explicitly out of scope.

## Protocol

| Component | Stage 3 policy |
| --- | --- |
| Dataset | M3FD, IR images first; visible paths are reported only |
| Backbone | Original UNIV student or teacher checkpoint branch |
| Features | `blocks2.1` spatial map and normalized `norm` tokens |
| Adapter | `MultiScaleTaskAdapter`, 256 output channels |
| Outputs | P3 `[B,256,28,28]`, P4 `[B,256,14,14]`, P5 `[B,256,7,7]` at 224² |
| Head | Minimal key/shape/finiteness validator (placeholder) |
| Goal | Prove downstream compatibility before implementing or training a detector |

The canonical class order is `people, car, bus, motorcycle, lamp, truck`.
Dataset inspection is read-only: it checks directories, class metadata, split
files, matching image/label stems, and reports missing or empty labels. It does
not convert annotations.

## Commands

Inspect a dataset tree (accepted directory aliases are included in the JSON):

```bash
python tools/check_m3fd_detection.py /path/to/M3FD
```

Run the student detector-boundary smoke test with a dummy image:

```bash
python tools/inspect_univ_mta_detection_smoke.py \
  --checkpoint /path/to/checkpoint0400.pth --checkpoint-key student --device cuda
```

Use one real IR image by adding `--image /path/to/M3FD/Ir/00000.png`. Repeat
with `--checkpoint-key teacher` to validate the teacher stream. A missing
checkpoint or a zero/insufficient checkpoint load fails before the compatibility
result; random UNIV weights are never accepted. The command only prints a JSON
report and never starts training.

## Exit criterion

Proceed to a real single-stream detector baseline only after all dataset checks
pass, checkpoint validation reports a nonzero acceptable load fraction, selected
features and P3/P4/P5 are finite, expected shapes match, and the placeholder head
reports `compatible: true`.
