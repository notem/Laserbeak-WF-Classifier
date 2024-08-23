# Laserbeak: Evolving Website Fingerprinting Attacks with Attention and Multi-Channel Feature Representation
### An improved website fingerprinting classification model that achieves state-of-the-art against defended network traffic


LASERBEAK is a state-of-the-art website fingerprinting (WF) attack model that achieves significant performance improvements against defended Tor traffic. By integrating multi-channel traffic representations and advanced techniques adapted from computer vision models, LASERBEAK pushes the boundaries of WF attack performance. Key innovations include:

- **Multi-Channel Feature Representation:** Provides a richer context for the model by leveraging various traffic features, improving robustness against defenses.
- **Advanced Transformer-Based Architecture:** Utilizes multi-headed attention layers to capture both local and global patterns in the traffic data.

In evaluations, LASERBEAK outperforms existing WF attacks, demonstrating up to 36.2% performance improvements in defended scenarios.

## Key Contributions

1. **Multi-Channel Traffic Representations:** Combining multiple traffic features leads to significant accuracy improvements against defended samples.
2. **Novel WF Model:** Leveraging state-of-the-art transformer-based techniques to enhance WF attack performance.
3. **Precision Optimizers:** Introducing new methods for improving precision in open-world scenarios, critical for realistic attacks.
4. **Comprehensive Dataset Evaluation:** Extensive testing on both closed-world and open-world datasets to validate the model's efficacy.

## Usage

### Training a LASERBEAK Model

To train a LASERBEAK model on a specified dataset, use the following command:

```bash
python benchmark.py \
    --data_dir /path/to/data \
    --ckpt_dir /path/to/checkpoints \
    --results_dir /path/to/results \
    --config ./configs/laserbeak.json \
    --dataset be-front \
    --epochs 20 \
    --multisamples 10 \
    --exp_name my_experiment
```

This command trains the model on the `be-front` dataset using x10 simulated samples for 20 epochs.

### Evaluation

To evaluate a pre-trained LASERBEAK model, use:

```bash
python benchmark.py \
    --data_dir /path/to/data \
    --ckpt /path/to/model_checkpoint.pth \
    --eval --dataset be-front
```

This command loads the model checkpoint and evaluates its performance on the `be-front` dataset.

## Available Datasets

The `benchmark.py` script supports various datasets for training and evaluation, including:

- `be`: Basic dataset
- `be-front`: Front-defended dataset
- `amazon`: Amazon traffic dataset
- `webmd`: WebMD traffic dataset
- `gong`: Gong dataset with Surakav defenses

Refer to the `--dataset` option in the command-line arguments for the full list of available datasets.

We provide these datasets as pre-packaged pickle files that can be downloaded from the this [Google drive directory](https://drive.google.com/drive/folders/1cRIujmDFUpVD0rA0U92bxeGaq5DMlzFm?usp=drive_link). 
To load the provided files using the `benchmark.py` script, the `--data_dir` argument should be used to direct to the parent directory containing the `wf-*` subdirectories. The file and subdirectory names must be the same as provided in the Google drive to function correctly.

## Warnings

**This code is research-grade and may contain bugs or other issues.** Users are advised to carefully validate results and use the code at their own risk.