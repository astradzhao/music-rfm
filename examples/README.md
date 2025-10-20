# MusicRFM Examples

This directory contains examples demonstrating how to use MusicRFM for controllable music generation.

## Quick Start

Run the examples in order:

### 1. Train a Control Direction

Train a direction to control a specific musical note:

```bash
python 01_train_note_direction.py
```

This will:
- Load the MusicGen model and Syntheory dataset
- Train an RFM direction to control note "C" vs other notes
- Save the trained direction to `./trained_directions/`

```bash
python 01_train_interval_direction.py
```
This will do the same on intervals dataset (tritone, interval 6).

### 2. Generate Controlled Music

Use the trained direction to generate music:

```bash
python 02_generate_controlled.py
```

This will:
- Load the trained direction from step 1
- Generate music with different control strengths
- Save baseline and controlled audio to `./generated_audio/`

**Features/hyperparams available for reducing audio artifacts or changing generation:**
- **Layer pruning**: Choose which layers to control (all, top-k, exponential dropout)
- **Time-varying control**: Apply exponential/linear decay over time
- **Probabilistic injection**: Control probability at each generation step
- **Custom coefficients**: Use different control strengths for regression vs classification

Edit the configuration section in the file to customize:
```python
LAYER_SELECTION = "all"  # Options: "all", "top_k", "exp_weighting"
TIME_CONTROL = None      # Options: None, "exp_decay", "linear_decay"
INJECT_CHANCE = 1.0      # Range: 0.0 to 1.0
```
Default generation config used in our paper is layer pruning = exp weighting (with base 1.0, coef 0.95), with inject chance 0.3


### 3. Temporal Control

Apply time-varying control during generation:

```bash
python 03_temporal_control.py
```
Functions included:
- Constant control (baseline)
- Exponentially decaying control
- Linearly increasing control

You can make your own functions easily! 

### 4. Multi-Direction Control (Optional)

Control multiple concepts simultaneously:

```bash
python 04_multidirection_control.py
```

**Note:** You'll need to train multiple concepts first (e.g., run step 1 with different notes or intervals, etc)


### 5. Train and Generate with Regression Direction (Optional)

Train a direction to control tempo (BPM) as a continuous value and generate controlled music:

```bash
# Train the tempo direction
python 05_regression_direction.py train

# Generate music with tempo control
python 05_regression_direction.py generate

# Or run both sequentially (default)
python 05_regression_direction.py
```

This will:
- **Training**: Load Syntheory tempos dataset, train RFM regression direction, save to `./trained_concepts/`
- **Generation**: Generate music with different tempo controls (slower/faster). Negative coef = slower, positive = faster.

It is actually recommended to train tempos with no pooling (using last token activations)

### 6. Train a Multiclass Direction

Train a single direction to distinguish between all musical notes (C, C#, D, etc.):

```bash
python 06_multiclass_direction.py
```
This will:
- Load the Syntheory notes dataset
- Train an RFM multiclass direction using one-hot encoding
- Save the trained direction and class metadata to `./trained_directions/`

Performing this with hyperparam sweep over n_components will yield results seen in our multiclass classification section of the paper.

For some categories with a large amount of data points, you may have to decrease the amount of training samples used depending on your GPU memory.


## Advanced Generation Options explained
### Layer Pruning Methods

**All Layers (default)**
```python
LAYER_SELECTION = "all"
```
- Controls all 47 decoder layers equally
- Best for maximum control strength
- Use when you want the strongest effect

**Top-K Selection**
```python
LAYER_SELECTION = "top_k"
TOP_K = 16
```
- Selects the k best-performing layers based on training results

**Exponential Dropout**
```python
LAYER_SELECTION = "exp_dropout"
EXP_BASE_WEIGHT = 1.0
EXP_DECAY_RATE = 0.95
```
- Uses all layers but with performance-weighted contributions

### Time-Varying Control

**Constant (default)**
```python
TIME_CONTROL = None
```
- Control coefficient stays constant throughout generation

**Exponential Decay**
```python
TIME_CONTROL = "exp_decay"
TIME_DECAY_RATE = 0.998
```
- Control gradually decreases over time
- Useful for smooth transitions
- Lower decay_rate = faster decay

**Linear Decay**
```python
TIME_CONTROL = "linear_decay"
```
- Control decreases linearly over 1500 steps
- Predictable, uniform decrease

### Probabilistic Injection

```python
INJECT_CHANCE = 0.3  # 30% probability
```
- Controls the probability of applying the direction at each generation step
- `1.0` = always apply (deterministic)
- `0.3` = apply 30% of the time (stochastic)
- Lower values = subtler, more natural control, prevents oversteering

## Customization

### Training Your Own Concepts

Modify `01_train_note_direction.py` to train different concepts:

```python
# Change the target note
TARGET_NOTE = "G"  # or "F#", "D", etc.

# Or use a different dataset attribute
# dataset = load_dataset("meganwei/syntheory", "chords")
# TARGET_CHORD = "major"
```

### Using Different Models

All examples use `musicgen-large`, but you can use smaller models:

```python
music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium"  # or "musicgen-small"
).to(DEVICE)
```
We are working on adapting this framework to other models, such as MagentaRT and OpenAI/Jukebox.

## Questions?

Open an issue on GitHub.

