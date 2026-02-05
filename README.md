Technical Implementation & Engineering Challenges
Model Specifications

    Architecture: MobileNetV2 (Alpha 0.35)

    Input Dimensions: 160x160x3 (RGB)

    Final Validation Accuracy: 91.4% * Optimization Target: NXP i.MX RT1060 (Cortex-M7)

Engineering Challenges & Resolutions
1. Environment Stabilization & Headless Training

    Challenge: Encountered a critical asyncio library regression within the eIQ graphical environment (Python 3.10 incompatibility), preventing the use of the standard GUI for model generation.

    Resolution: Developed a custom Headless Training Pipeline using the TensorFlow Functional API. This allowed for direct environment isolation and manual control over the model graph, bypassing graphical dependencies.

2. Memory Constraint Optimization

    Challenge: The target hardware (NXP i.MX RT series) features limited internal SRAM (TCM). Standard MobileNet models often exceed this limit, forcing slow execution from external Flash.

    Resolution: Applied a 0.35 Alpha width multiplier. This hyperparameter adjustment reduced the number of filters per layer, resulting in a model footprint of ~630 KB. This ensures the entire model resides within the Tightly Coupled Memory (TCM), achieving ultra-low latency inference.

3. Advanced Regularization & Overfitting Control

    Challenge: Initial training iterations showed signs of "memorization" (overfitting) due to high consistency in the wafer defect dataset.

    Resolution: Integrated a 20% Dropout layer and implemented Transfer Learning using ImageNet-pretrained weights. This shifted the accuracy from a suspicious 100% to a robust 91%, indicating the model is now identifying generalized structural defects rather than specific pixel coordinates.

4. Multi-Channel Signal Preservation

    Challenge: Evaluating the trade-off between 1-channel (Grayscale) memory savings and 3-channel (RGB) feature depth.

    Resolution: Standardized on 3-channel RGB. While increasing the input buffer to 76.8 KB, it preserved critical chromatic signatures—such as surface oxidation and chemical discoloration—that are vital for high-fidelity wafer inspection.

Deployment Pipeline

    Preprocessing: Automated directory-based labeling for 5,270 images.

    Training: Custom Keras implementation with Dropout and Alpha scaling.

    Graph Conversion: Exported to ONNX (Opset 13) to ensure interoperability with NXP backend engines (GLOW/DeepViewRT).

    Quantization: Prepared for INT8 Post-Training Quantization to leverage hardware acceleration.

Summary of Results

The final model achieves a balanced 91% accuracy with an optimized footprint that fits the specific hardware constraints of the NXP i.MX RT1060, making it ready for real-time edge deployment.
