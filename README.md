Mixture of Experts (MoE) in PyTorch

A simple yet extensible Mixture of Experts implementation in PyTorch. This project demonstrates how to split model capacity across multiple “expert” networks, with a trainable router that dynamically selects which experts handle each input.

Features
	•	Configurable Experts: Easily change the number or structure of experts (e.g., linear layers, MLPs).
	•	Top-K Routing: Only the most relevant experts per input are activated, reducing compute.
	•	Batched Forward: Efficiently processes all inputs for each expert together, avoiding slow per-sample calls.

Requirements
	•	Python 3.7+ (or higher)
	•	PyTorch (tested on 1.13+ but should work with similar versions)

