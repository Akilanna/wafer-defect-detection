## Technical Implementation: Wafer Defect Inspection

### Optimized Inference Specifications
* **Architecture**: MobileNetV2 (Alpha 0.35)
* **Input Resolution**: 160x160x3 (Optimized for Edge Performance)
* **Accuracy**: 87.57% (Hardware-Validated)
* **Model Footprint**: 627.7 KB (INT8 Quantized)
* **Inference Engine**: NXP eIQ DeepViewRT / GLOW

### Engineering Highlights
1. **TCM Residency Optimization**: Achieved a 627.7 KB footprint via INT8 Post-Training Quantization (PTQ). This ensures the model resides entirely within Tightly Coupled Memory (TCM), bypassing external Flash latency for deterministic real-time performance.
2. **Channel Broadcasting Strategy**: Implemented a channel broadcasting pipeline to adapt 1-channel SEM grayscale data to the optimized 3-channel MobileNet backbone without compromising validated accuracy.
3. **Toolchain Resilience**: Resolved metadata mismatches and environment library regressions within the eIQ toolkit to finalize the hardware-ready deployment.

## Project Evolution & Engineering Resilience
Our team encountered several critical hurdles during the development of this inspection system:
* **Environment Stability**: Resolved eIQ Toolkit dependency failures via source-level patching.
* **Metadata Reconciliation**: Overcame ONNX shape-lock conflicts using custom Python shims.
* **Hardware Constraints**: Prioritized INT8 quantization to meet strict TCM residency requirements for sub-millisecond inference.
