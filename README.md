# Pruning a ResNet-18-like Network on CIFAR-100

Lab 1 of *Advanced Topics in Image Processing and Computer Vision* (Université de Bordeaux, 2026–2027) — Gourab Roy.

I trained a compact ResNet-18-like network on CIFAR-100 and pruned it with the `torch.nn.utils.prune` package of PyTorch:
layer-by-layer and global L1 pruning, re-training of the pruned models, iterative pruning, and (bonus) my own structured
filter-pruning method. The full report is in [`report/Gourab_Roy_Model_Pruning.pdf`](report/Gourab_Roy_Model_Pruning.pdf).

## Main results

| Model | Sparsity | Validation acc. | Test acc. |
|---|---:|---:|---:|
| Baseline (Exercise 0) | 0% | 68.67% | 68.54% |
| Global L1 pruning, no re-training (Exercise 3) | 50% | 67.67% | 67.37% |
| **Global L1 pruning + 5 epochs of re-training (Exercise 4) — final model** | **70%** | **68.05%** | **68.34%** |
| Iterative pruning + re-training (Exercise 5) | 90% | 66.03% | 66.09% |
| Structured filter pruning + re-training (Bonus) | 20% of filters | 67.76% | – |

Layer-by-layer pruning already drops to 26.73% at 20% sparsity, while global pruning keeps the accuracy within one point
up to 50% without re-training. With five epochs of re-training, 70% of the weights can be removed for only 0.20 points of
test accuracy.

## Repository structure

```
├── src/model.py              ResNet-18-like network (half width, no max-pooling in the stem)
├── train.py                  Exercise 0: baseline training (saves best_model_params.pt and results/exercise0/)
├── pruning.py                Exercises 1–5 and Bonus (choose with run_exercises at the top of the file)
├── evaluate.py               Independent evaluation of any saved model (validation and test sets)
├── requirements.txt          Package versions used
├── cifar100_split.pt         Saved 40,000 / 10,000 train/validation split (seed 42)
├── best_model_params.pt      Trained baseline (68.67% validation)
├── results/                  Per exercise: CSV results, logs, figures and an answers file (*_answers.md)
│   ├── exercise0 … exercise5
│   ├── final_evaluation
│   └── bonus
└── report/                   LaTeX source and PDF of the report
    ├── main.tex                    LaTeX source of the report
    ├── Gourab_Roy_Model_Pruning.pdf   compiled report
    ├── figures/              Figures exported from the experiments
    ├── pgf_figures/          PGFPlots figures (drawn in LaTeX from report/data)
    └── data/                 Data of the PGFPlots figures + prepare_figure_data.py
```

## Running the code

```bash
# Environment (Python 3.13, CUDA 12.6 wheels for PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

python train.py        # Exercise 0 (~9 min on an RTX 4060 laptop GPU); CIFAR-100 is downloaded to data/
python pruning.py      # set run_exercises = [1], [2], [3], [4], [5] or ["bonus"] at the top of the file
python evaluate.py     # final evaluation of the selected models on validation and test sets
python evaluate.py results/exercise4/exercise4_finetuned_70.pt --split test   # evaluate a single model
```

Exercise 4 re-trains the models saved by Exercise 3, so run the exercises in order.

## About the saved models

To keep the repository small, only four models are included: the baseline (`best_model_params.pt`), the Exercise 1 model
(`results/exercise1/exercise1_pruned_20.pt`), the final model (`results/exercise4/exercise4_finetuned_70.pt`) and the kept
bonus model (`results/bonus/bonus_structured_20.pt`). All the other pruned models (≈ 950 MB in total) are re-created by
running `pruning.py`. The CIFAR-100 dataset is not included; it is downloaded automatically by the scripts.

## Building the report

```bash
cd report
pdflatex -jobname=Gourab_Roy_Model_Pruning main.tex && pdflatex -jobname=Gourab_Roy_Model_Pruning main.tex
```

The PGFPlots figures read `report/data/*.csv`, which can be regenerated from the results with
`python report/data/prepare_figure_data.py` (run from the repository root).
