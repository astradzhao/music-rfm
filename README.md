# MusicRFM: Recursive Feature Machines for Music Generation Control

MusicRFM is a Python library for fine-grained controllable music generation using Recursive Feature Machines (RFM). It enables you to train interpretable directions in the latent space of music generation models (like MusicGen) and use them to control specific musical attributes during generation.

## Features

- 🎵 **Fine-grained Control**: Control specific musical attributes like notes, chords, scales, tempo, and more
- **Interpretable Directions**: Learn interpretable control vectors using RFM and other methods
- **Temporal Control**: Apply time-varying control during generation (e.g., crossfading between concepts)
- **Multi-Concept Control**: Combine multiple control directions simultaneously
- Supports RFMs and linear probing for multiclass classification

## Installation

```bash
# Clone the repository
git clone https://github.com/astradzhao/music-rfm.git
cd music-rfm

# Install in development mode
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch
- Transformers (for MusicGen models)
- xrfm (for RFM algorithm)
- See `requirements.txt` for full list

## Quick Start

### 1. Train a Control Direction

```python
from musicrfm import MusicGenController
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
import torch

# Load models
device = "cuda"
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large").to(device)
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
encodec_model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

# Create controller
controller = MusicGenController(
    music_model,
    music_processor,
    encodec_model,
    encodec_processor,
    control_method="music_rfm",
    n_components=12,
    rfm_iters=30,
    batch_size=16
)

# Train directions using your labeled data
# train_data: audio features (tensor)
# train_labels: labels for the concept you want to control (tensor)
controller.compute_directions(
    train_data=train_data,
    train_labels=train_labels,
    test_data=test_data,
    test_labels=test_labels,
    hidden_layers=list(range(-1, -48, -1)),  # Which model layers to compute directions for
    tuning_metric='auc'
)

# Save the learned directions
controller.save(
    concept="my_concept",
    model_name="musicgen_large",
    path="./directions"
)

```
For customizable control, you will also need to the results file. Please see `01_train_note_direction.py` for an example.

### 2. Generate Controlled Music

```python
# Load previously trained directions
controller.load(
    concept="my_concept",
    model_name="musicgen_large",
    path="./directions"
)

# Generate music with control
prompts = ["A relaxing jazz song with piano"]
layers_to_control = list(range(-1, -48, -1))  # Control all layers
control_coef = 0.6  # Strength of control

controlled_audio = controller.generate(
    prompts,
    layers_to_control=layers_to_control,
    control_coef=control_coef,
    max_new_tokens=1500
)

# Save the audio
import soundfile as sf
sf.write("output.flac", controlled_audio[0].cpu().numpy(), 32000, format="FLAC")
```

## Examples and Advanced Features

Check out the `examples/` directory and its `README.md` for complete working examples. These may provide better insight on the pipeline. The README also shows advanced features such as temporal control, multidirection control, and layer pruning.

## Citation

If you use MusicRFM in your research, please cite:

```bibtex
@article{zhaosteering2025,
  title={Steering Autoregressive Music Generation with Recursive Feature Machines},
  author={Daniel Zhao, Daniel Beaglehole, Taylor Berg-Kirkpatrick, Julian McAuley, Zachary Novack},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of [MusicGen](https://github.com/facebookresearch/audiocraft) by Meta AI
- Uses the [xRFM](https://github.com/dmbeaglehole/xRFM) library for RFMs (Thanks Daniel!)
