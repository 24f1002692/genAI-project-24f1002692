# Smart MCQ Solver Challenge

This project builds a pipeline that reads a multiple-choice question (MCQ) with 5 options (A–E) and predicts the top-3 most likely correct answers. The predictions are scored using the **MAP@3** metric, as required by the Kaggle competition.

The project moves through five milestones, from simple text similarity methods to a final custom-built (from scratch) deep learning model:

| Milestone | Approach | Notes |
|---|---|---|
| M1 | TF-IDF / Word2Vec + cosine similarity | Classical NLP baseline |
| M2 | BERT / RoBERTa / SBERT / Zero-shot (BART-MNLI) | Pretrained, no fine-tuning |
| M3 | RAG (FAISS + Flan-T5) | Adds Wikipedia knowledge |
| M4 | LoRA fine-tuned BERT | Task-adapted pretrained model |
| M5 | DeBERTa + RoBERTa ensemble | Weighted ensemble (0.70 / 0.30) |
| Final | BiGRU (from scratch) + RAG context | Best model — Val MAP@3: 0.996, Leaderboard: ~0.75 |

The final model is a **Bidirectional GRU (BiGRU) with attention pooling**, trained completely from scratch (no pretrained weights), and uses extra context retrieved from Wikipedia (RAG) to make up for the small training dataset.

---

## Project Structure

This repo has **two branches** — pick the one you need:

- **`main`** — the final, cleaned-up project (all trained models, source code, and the deployed app)
- **`master`** — has a `notebooks/milestones` folder with the notebooks for each milestone (M1–M5) shown in the report

### `main` branch

```
genAI-project-24f1002692/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── smart-mcq-solver-challenge.zip
├── notebooks/
│   ├── models/                        # one notebook per model (e.g. LoRA_MC.ipynb, BiGRU_Rag.ipynb)
│   ├── EDA.ipynb                      # exploratory data analysis notebook
│   └── eda_mcq.png                    # EDA chart used in the report
├── outputs/
│   ├── BiGRU_Rag/                     # final model checkpoint, vocab, training curves
│   └── LoRA_MC/                       # LoRA fine-tuned model outputs
├── src/
│   ├── dataset/                       # data loading & preprocessing code
│   ├── inference/                     # inference / prediction code
│   ├── models/                        # model definitions (e.g. BiGRU OptionScorer)
│   └── __init__.py
├── DL-24f1002692-notebook-t22026.ipynb  # main Kaggle submission notebook (BiGRU + RAG, production model)
├── app.py                             # deployed app (loads the trained model and serves predictions)
├── requirements.txt                   # dependencies for training/notebooks
├── requirements-deploy.txt            # dependencies for running app.py
├── .gitignore
└── README.md
```

### `master` branch

```
genAI-project-24f1002692/
└── notebooks/
    └── milestones/
        ├── M1_TFIDF_Word2Vec.ipynb          # (naming may vary — see folder for exact files)
        ├── M2_Pretrained_Transformers.ipynb
        ├── M3_RAG.ipynb
        ├── M4_LoRA_FineTuning.ipynb
        └── M5_Ensemble.ipynb
```

To switch between branches after cloning:

```bash
git checkout main     # final models, src code, deployed app
git checkout master   # milestone-wise notebooks (M1–M5)
```

---

## 1. Requirements

Before starting, make sure you have:

- A **GitHub** account (to clone the repo)
- A **Kaggle** account with an API token (`kaggle.json`) — needed to download the Wikipedia dataset used for RAG
- **Python 3.10+**
- A machine with a **GPU** is strongly recommended (training and RAG search are slow on CPU)

---

## 2. Running the Project Locally (On Your Own Computer)

### Step 1: Clone the repository

```bash
git clone https://github.com/24f1002692/genAI-project-24f1002692.git
cd genAI-project-24f1002692
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install the required packages

```bash
pip install -r requirements.txt --quiet
pip install faiss-gpu-cu12 sentence-transformers peft torchao pyarrow fastparquet kaggle wandb --quiet
```

> If you don't have a GPU, install `faiss-cpu` instead of `faiss-gpu-cu12`.

> If you see a numpy version conflict warning, you can usually ignore it unless you actually get an import error. If `sklearn` fails to import, run:
> ```bash
> pip install "numpy<2.0" --quiet
> ```

### Step 4: Set up your Kaggle API key

1. Go to your Kaggle account settings and create a new API token — this downloads a file called `kaggle.json`.
2. Place this file in a folder named `.kaggle` in your home directory.
3. Set the correct permissions (Mac/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 5: Download the Wikipedia dataset (needed for the RAG models)

```bash
mkdir -p wiki_faiss wiki_text
kaggle datasets download -d jjinho/wikipedia-2023-07-faiss-index -p wiki_faiss --unzip
kaggle datasets download -d jjinho/wikipedia-20230701 -p wiki_text --unzip
```

> This is a large download (~18 GB), so it will take some time. Do **not** upload these folders to GitHub — they are too large and can always be downloaded again.

### Step 6: Point the code to your dataset folders

At the top of the notebook you want to run, add:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

os.environ["WIKI_FAISS_INDEX_DIR"] = os.path.abspath("wiki_faiss")
os.environ["WIKI_TEXT_DIR"] = os.path.abspath("wiki_text")
```

This tells the code where to find the shared helper code (`src/`) and the Wikipedia data.

### Step 7: Open and run the notebook

Open the project folder in VS Code or Jupyter.

- For the **final model**, open `DL-24f1002692-notebook-t22026.ipynb` (this is on the `main` branch).
- For the **EDA**, open `notebooks/EDA.ipynb`.
- For the **milestone-wise notebooks (M1–M5)**, switch to the `master` branch first (`git checkout master`), then open the notebooks inside `notebooks/milestones/`.

Then run all the cells.

```bash
jupyter notebook
```

or open the folder directly in VS Code and run the notebook from there.

---

## 3. Running the Project on Lightning AI Studio (Cloud, with Free GPU)

If you don't have a GPU on your own computer, you can run this project for free on Lightning AI Studio.

### Step 1: Create a Studio

1. Go to [lightning.ai](https://lightning.ai) and create a new Studio.
2. On the "Choose how you want to build" screen, select **Advanced**.
3. For the environment, select **Browser** (this gives you VS Code in your browser).
4. Under Git integration, select **GitHub** and connect/authorize your account.
5. Once the Studio opens, click the machine selector (top-right) and select a **GPU (T4)** machine.

### Step 2: Clone the repository

Open a terminal (`Terminal → New Terminal`) and run:

```bash
git clone https://github.com/24f1002692/genAI-project-24f1002692.git
cd genAI-project-24f1002692
```

Since GitHub is already connected to the Studio, no extra login is needed.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt --quiet
pip install faiss-gpu-cu12 sentence-transformers peft torchao pyarrow fastparquet kaggle wandb --quiet
```

### Step 4: Set up Kaggle and download the dataset

1. In the file explorer, create a `.kaggle` folder (if it doesn't exist) and upload your `kaggle.json` file into it (right-click → Upload).
2. Set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download the dataset:
   ```bash
   mkdir -p ~/wiki_faiss ~/wiki_text
   kaggle datasets download -d jjinho/wikipedia-2023-07-faiss-index -p ~/wiki_faiss --unzip
   kaggle datasets download -d jjinho/wikipedia-20230701 -p ~/wiki_text --unzip
   ```
4. Check that the files downloaded correctly:
   ```bash
   ls /teamspace/studios/this_studio/wiki_faiss
   ls /teamspace/studios/this_studio/wiki_text
   ```

### Step 5: Open the notebook

In the file explorer, open `DL-24f1002692-notebook-t22026.ipynb` (the final model, on the `main` branch) — or, if you want the milestone-wise notebooks (M1–M5), switch to the `master` branch first (`git checkout master`) and open the notebooks inside `notebooks/milestones/`. Click "Select Kernel" (top-right) and choose your Python environment.

### Step 6: Add the setup cell

At the top of the notebook, add this cell and run it first:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

os.environ["WIKI_FAISS_INDEX_DIR"] = "/teamspace/studios/this_studio/wiki_faiss"
os.environ["WIKI_TEXT_DIR"] = "/teamspace/studios/this_studio/wiki_text"
```

### Step 7: Move the FAISS index to the GPU (makes RAG search much faster)

In the cell where the data is loaded, right after the index is loaded, add:

```python
import faiss
res = faiss.StandardGpuResources()
wiki_index = faiss.index_cpu_to_gpu(res, 0, wiki_index)
```

> If you skip this step, the search will run on the CPU and be much slower.

### Step 8: Run the notebook

Click **"Run All"** in the top toolbar, or run the cells one at a time (▶ or Shift+Enter).

To check that the GPU is being used, open a new terminal tab and run:

```bash
nvidia-smi
```

---

## 4. Saving and Organizing Your Trained Models

After training a model, create a folder for it inside `outputs/` (matching the pattern already used in the repo — `outputs/BiGRU_Rag/`, `outputs/LoRA_MC/`, etc.):

```bash
mkdir -p outputs/BiGRU_Rag
```

Move the checkpoint, vocab, training curves, and submission file into that folder, for example:

```bash
mv best_scratch_rag_model.pt outputs/BiGRU_Rag/
mv training_curves.png outputs/BiGRU_Rag/
mv submission_scratch_rag.csv outputs/BiGRU_Rag/
```

If you're working on the `master` branch with the milestone notebooks (M1–M5), keep each milestone's outputs next to its notebook inside `notebooks/milestones/` so it's clear which output belongs to which milestone.

---

## 5. Pushing Your Work to GitHub

### One-time setup: `.gitignore`

```bash
echo "outputs/*/submission_*.csv" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

This makes sure that:
- Submission CSV files are not pushed
- Python cache files are not pushed
- The Wikipedia dataset folders (kept outside the project folder) are never pushed by accident

### Push your changes

```bash
git add outputs/BiGRU_Rag/ DL-24f1002692-notebook-t22026.ipynb .gitignore
git status   # check that only the right files are staged
git commit -m "Add trained BiGRU-RAG model checkpoint and training curves"
git push origin main
```

If you're pushing milestone notebook changes, make sure you're on the `master` branch first:

```bash
git checkout master
git add notebooks/milestones/
git commit -m "Update milestone notebooks"
git push origin master
```

---

## 6. Deployment (Optional)

The `main` branch also includes a deployed app (`app.py`) that loads the trained model and serves predictions.

```bash
pip install -r requirements-deploy.txt --quiet
python app.py
```

> `requirements-deploy.txt` only has the packages needed to *run* the app (lighter than `requirements.txt`, which is for training/notebooks).

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Add the `sys.path.append` cell shown in Step 6 |
| `ModuleNotFoundError: No module named 'faiss'` / `'pyarrow'` etc. | Run `pip install <package> --quiet`, then restart the kernel |
| `RuntimeError: could not open .../wikipedia_202307.index` | Check that the environment variables are set correctly and the dataset was downloaded |
| `GPU-Util 0%` or FAISS search is very slow | Install `faiss-gpu-cu12` and move the index to the GPU (see Step 7 in the Lightning AI section) |
| numpy version conflict warnings | Safe to ignore unless you get an actual import error |
| Pasting into the terminal doesn't work (Lightning AI) | Try `Ctrl+Shift+V`, or type the command directly |

---

## 8. Results Summary

| Metric | Score |
|---|---|
| Validation MAP@3 (final BiGRU-RAG model) | 0.996 |
| Public leaderboard MAP@3 | ≈ 0.75 |
| Competition qualification cutoff | 0.73 |

The validation score is much higher than the leaderboard score. This is expected — it points to a difference between the style/topics of the training questions and the hidden test questions, not to a data leakage problem (this was checked carefully during the project).

---

**Author:** Shivam | Student ID: 24f1002692
**Course:** DL / GenAI, Term T22026