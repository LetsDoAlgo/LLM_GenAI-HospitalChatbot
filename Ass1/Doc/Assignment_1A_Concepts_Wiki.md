# Assignment 1A — Concepts Wiki
## Continual Pre-Training (CPT) for Domain Adaptation

> A comprehensive reference of all key concepts, techniques, and terminology used in the CPT pipeline notebook.

---

## Table of Contents

1. [Continual Pre-Training (CPT)](#1-continual-pre-training-cpt)
2. [BioGPT — The Base Model](#2-biogpt--the-base-model)
3. [Data Collection & PDF Extraction](#3-data-collection--pdf-extraction)
4. [Text Cleaning Pipeline](#4-text-cleaning-pipeline)
   - 4.1 [Length Filtering](#41-length-filtering)
   - 4.2 [Near-Duplicate Detection (MinHash LSH)](#42-near-duplicate-detection-minhash-lsh)
   - 4.3 [Language Filtering](#43-language-filtering)
5. [Tokenization](#5-tokenization)
   - 5.1 [BPE (Byte Pair Encoding)](#51-bpe-byte-pair-encoding)
   - 5.2 [Special Tokens (BOS / EOS / PAD)](#52-special-tokens-bos--eos--pad)
6. [Sequence Packing](#6-sequence-packing)
7. [Model Architecture Concepts](#7-model-architecture-concepts)
   - 7.1 [Causal Language Modeling (CLM)](#71-causal-language-modeling-clm)
   - 7.2 [lm_head (Language Modeling Head)](#72-lm_head-language-modeling-head)
   - 7.3 [Gradient Checkpointing](#73-gradient-checkpointing)
   - 7.4 [Mixed Precision (bf16 / fp16)](#74-mixed-precision-bf16--fp16)
8. [Training Concepts](#8-training-concepts)
   - 8.1 [Cross-Entropy Loss](#81-cross-entropy-loss)
   - 8.2 [Learning Rate & Cosine Scheduler](#82-learning-rate--cosine-scheduler)
   - 8.3 [Warmup Steps](#83-warmup-steps)
   - 8.4 [Gradient Accumulation](#84-gradient-accumulation)
   - 8.5 [Weight Decay (L2 Regularization)](#85-weight-decay-l2-regularization)
   - 8.6 [DataCollatorForLanguageModeling](#86-datacollatorforlanguagemodeling)
   - 8.7 [TrainerCallback](#87-trainercallback)
9. [Evaluation Concepts](#9-evaluation-concepts)
   - 9.1 [Perplexity (PPL)](#91-perplexity-ppl)
   - 9.2 [Catastrophic Forgetting](#92-catastrophic-forgetting)
10. [Inference & Text Generation](#10-inference--text-generation)
    - 10.1 [Nucleus Sampling (top-p)](#101-nucleus-sampling-top-p)
    - 10.2 [Top-k Sampling](#102-top-k-sampling)
    - 10.3 [Temperature](#103-temperature)
    - 10.4 [Repetition Penalty](#104-repetition-penalty)
11. [File Formats & Storage](#11-file-formats--storage)

---

## 1. Continual Pre-Training (CPT)

**What:** Taking a model that was already pre-trained on general data and training it further on domain-specific data.

**Why:** The base model (BioGPT) knows general biomedical language from PubMed, but lacks knowledge of specific clinical protocols, drug guidelines, and treatment pathways. CPT injects this specialized knowledge.

**How it differs from fine-tuning:**

| | Pre-Training | Continual Pre-Training (CPT) | Fine-Tuning (SFT) |
|---|---|---|---|
| **Data** | Massive general corpus | Domain-specific unlabeled text | Task-specific instruction pairs |
| **Objective** | Next-token prediction | Next-token prediction (same) | Next-token prediction on Q&A pairs |
| **Goal** | Learn language | Learn domain vocabulary & patterns | Learn to follow instructions |
| **Scale** | Trillions of tokens | Thousands–millions of tokens | Hundreds–thousands of examples |

**Key risk:** If the learning rate is too high or training runs too long, the model may forget general knowledge (see [Catastrophic Forgetting](#92-catastrophic-forgetting)).

---

## 2. BioGPT — The Base Model

- **Full name:** Biomedical Generative Pre-trained Transformer
- **Model ID:** `microsoft/biogpt`
- **Parameters:** ~347 million
- **Architecture:** GPT-style decoder-only transformer
- **Pre-trained on:** 15 million PubMed abstracts
- **Type:** Causal (autoregressive) language model — generates text left-to-right

**Architecture details (from the notebook):**

| Property | Value |
|----------|-------|
| Decoder layers | 24 |
| Attention heads | 16 |
| Hidden size | 1024 |
| Head dimension | 64 (= 1024 / 16) |
| Vocabulary size | 42,384 |
| Max position embeddings | 1024 |

**Why BioGPT for clinical CPT:** It already understands biomedical terminology, so CPT builds on existing knowledge rather than starting from scratch.

---

## 3. Data Collection & PDF Extraction

### Sources Used
- **WHO** (World Health Organization) — Clinical guidelines via `iris.who.int`
- **NICE** (National Institute for Health and Care Excellence, UK) — `nice.org.uk`

### PyMuPDF (`fitz`)
A Python library for extracting text from PDF files page-by-page. Imported as `fitz` for historical reasons (it wraps the MuPDF C library).

```python
import fitz
doc = fitz.open("file.pdf")
text = doc[0].get_text("text")  # Extract plain text from page 0
```

**Limitations:** Cannot extract text from scanned/image PDFs; loses table formatting.

---

## 4. Text Cleaning Pipeline

The pipeline applies three sequential filters to improve corpus quality.

### 4.1 Length Filtering

**What:** Remove documents shorter than a threshold (200 characters in this notebook).

**Why:** Short extracted texts are usually cover pages, tables of contents, or reference-only appendices — they add noise and no meaningful clinical content.

### 4.2 Near-Duplicate Detection (MinHash LSH)

**Problem:** Clinical guidelines from different organizations often share boilerplate text (disclaimers, methodology sections). Training on duplicates wastes compute and can bias the model.

#### Shingles (k-shingles)

Overlapping sequences of *k* consecutive words used as the comparison unit for similarity.

**Example** (3-word shingles from `"the patient was treated with antibiotics"`):
```
"the patient was"
"patient was treated"
"was treated with"
"treated with antibiotics"
```

- **Why 3 words?** 1-word is too common (false matches); 5+ is too strict (misses paraphrases).
- **Shingles vs n-grams:** Same extraction, different usage. Shingles → set-based similarity. N-grams → frequency/probability models.

#### MinHash

A technique to compress a document's shingle set into a fixed-size **signature** (128 hash values in this notebook). Two documents with similar signatures share many of the same shingles.

**How it works:**
1. Collect all 3-word shingles from a document → a set
2. Apply multiple hash functions to each shingle
3. Keep the minimum hash value for each function → the signature
4. Compare signatures: fraction of matching positions ≈ Jaccard similarity

#### LSH (Locality-Sensitive Hashing)

An indexing structure that groups similar MinHash signatures into the same "bucket." This allows O(1) lookups instead of comparing every pair of documents (which would be O(n²)).

**Threshold:** 0.8 in this notebook — documents sharing ≥80% of their 3-word shingles are considered duplicates.

### 4.3 Language Filtering

**What:** Use the `langdetect` library to identify the language of each document and keep only English ones.

**Why:** Some WHO documents contain multilingual sections. Non-English text would confuse BioGPT's English-only tokenizer.

**Implementation:** Only the first 1000 characters are checked (faster and usually sufficient).

---

## 5. Tokenization

### 5.1 BPE (Byte Pair Encoding)

BioGPT uses a **Moses tokenizer + BPE** combination:

1. **Moses** first splits text into words using rule-based tokenization
2. **BPE** then splits words into subword units based on frequency

**Example:**
```
"antiretroviral" → ["anti", "retro", "vir", "al"]
```

**Why subwords?** Handles rare/unseen words by decomposing them into known pieces. A 42,384-token vocabulary can represent any text.

### 5.2 Special Tokens (BOS / EOS / PAD)

| Token | Meaning | Purpose |
|-------|---------|---------|
| **BOS** (Beginning of Sequence) | `</s>` in BioGPT | Signals the start of a document |
| **EOS** (End of Sequence) | `</s>` in BioGPT | Signals the end of a document |
| **PAD** (Padding) | Set to `</s>` | Fills unused positions in fixed-length batches |

**Note:** BioGPT uses `</s>` for both BOS and EOS. GPT-style models often lack a dedicated PAD token, so EOS is reused.

---

## 6. Sequence Packing

**Problem:** Documents vary in length. Without packing, short documents are padded to `max_length` with PAD tokens, wasting GPU compute.

**Solution — Packing:**
1. Tokenize all documents
2. Wrap each with `[BOS] ... [EOS]`
3. Concatenate into one flat token stream
4. Slice into fixed-length chunks (1024 tokens each)

```
Non-Packed:
  [BOS] short_doc [EOS] [PAD] [PAD] [PAD] [PAD] [PAD]    ← 60% wasted
  [BOS] long_doc_that_fills_everything [EOS]               ← 0% wasted

Packed:
  [BOS] short_doc [EOS] [BOS] next_doc [EOS] [BOS] more..  ← 0% wasted
  ...tokens_continued [EOS] [BOS] another_doc [EOS] [BOS]  ← 0% wasted
```

**Benefits:**
- ~100% GPU utilization (no padding)
- BOS/EOS tokens inside the stream mark document boundaries, preventing the model from learning cross-document patterns

**Trade-off:** A single packed sequence may contain parts of 2+ documents. The model sees artificial transitions at document boundaries, but BOS/EOS tokens mitigate this.

---

## 7. Model Architecture Concepts

### 7.1 Causal Language Modeling (CLM)

The training objective where the model predicts the **next token** given all previous tokens. Each position can only attend to tokens to its left (causal masking).

```
Input:   The patient was treated with
Label:   patient was treated with antibiotics
```

The model learns P(next_token | all_previous_tokens).

### 7.2 lm_head (Language Modeling Head)

The final linear layer that projects the model's hidden states (size 1024) to vocabulary logits (size 42,384). Each position outputs a probability distribution over the entire vocabulary.

```
hidden_state [1024] → lm_head (Linear) → logits [42,384] → softmax → probabilities
```

**Verification:** The notebook checks that `lm_head.out_features == vocab_size` to ensure the model can predict over its full vocabulary.

### 7.3 Gradient Checkpointing

**Problem:** Storing all intermediate activations during forward pass consumes lots of VRAM.

**Solution:** Only store activations at certain "checkpoint" layers. During backward pass, recompute the missing activations on-the-fly.

**Trade-off:**
- Saves ~40% VRAM
- Costs ~20% more compute time (recomputation)
- Essential for training on GPUs with limited memory (e.g., T4 with 16 GB)

### 7.4 Mixed Precision (bf16 / fp16)

Using lower-precision floating point numbers instead of full 32-bit (fp32):

| Format | Bits | Range | Precision | Use Case |
|--------|------|-------|-----------|----------|
| fp32 | 32 | Large | High | CPU training |
| bf16 | 16 | Same as fp32 | Lower | Preferred on modern GPUs (A100, H100) |
| fp16 | 16 | Smaller | Lower | Fallback for older GPUs (T4, V100) |

**Benefits:** Halves memory usage, faster matrix operations. The model is stored in bf16/fp16 but critical operations (loss, gradients) use fp32 for numerical stability.

---

## 8. Training Concepts

### 8.1 Cross-Entropy Loss

The loss function for language modeling. Measures how different the model's predicted probability distribution is from the true next token.

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log P(t_i | t_1, ..., t_{i-1})$$

- Lower loss = model is more confident about the correct next token
- A random model over 42K vocab would have loss ≈ $\log(42384) ≈ 10.65$
- A pre-trained model starts at loss ≈ 2–4 on domain text

### 8.2 Learning Rate & Cosine Scheduler

**Learning Rate (LR):** Controls how much the model's weights change per update step. Used: `2e-5` (0.00002).

**Cosine Scheduler:** Gradually decays the LR following a cosine curve:

```
LR
 |  /‾‾‾‾\
 | /      \
 |/        \___________
 +-------------------------→ Steps
   warmup    cosine decay
```

**Why cosine?** Smoother than step decay; avoids sudden LR drops that can destabilize training.

### 8.3 Warmup Steps

The first N training steps where the LR gradually increases from 0 to the target value.

**Why:** Without warmup, the model receives large gradient updates on the first few batches (which are not representative of the full dataset), potentially destabilizing weights.

**Used:** 50 warmup steps (GPU) / 2 steps (CPU demo mode).

### 8.4 Gradient Accumulation

**Problem:** GPU memory limits the batch size (e.g., only 2 sequences fit).

**Solution:** Accumulate gradients over multiple forward passes before updating weights.

```
Effective batch size = per_device_batch_size × gradient_accumulation_steps
                     = 2 × 4 = 8
```

The model sees 8 sequences worth of signal per weight update, but only holds 2 in memory at a time.

### 8.5 Weight Decay (L2 Regularization)

Adds a penalty proportional to the magnitude of weights, preventing them from growing too large.

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{CE}} + \lambda \sum w_i^2$$

**Used:** `weight_decay=0.01` — mild regularization to prevent overfitting on the small clinical corpus.

### 8.6 DataCollatorForLanguageModeling

A HuggingFace utility that prepares batches for language modeling:
- For **Causal LM** (`mlm=False`): shifts `input_ids` right by one position to create `labels` (next-token targets)
- Handles padding within a batch

### 8.7 TrainerCallback

A hook system in HuggingFace's `Trainer` that lets you execute custom code at specific training events (on_log, on_epoch_end, on_save, etc.).

**Used in this notebook:** `LossLoggerCallback` records the training loss at each logging step for later visualization.

---

## 9. Evaluation Concepts

### 9.1 Perplexity (PPL)

A standard metric for language models. Measures how "surprised" the model is by the text.

$$\text{PPL} = e^{\text{average cross-entropy loss}}$$

| PPL Value | Interpretation |
|-----------|---------------|
| Lower | Model predicts the text well (confident and accurate) |
| Higher | Model is uncertain / unfamiliar with the text |
| 1.0 | Perfect prediction (impossible in practice) |
| ~42,384 | Random guessing over the vocabulary |

**In this notebook:**
- **Base model PPL** on clinical text → how well BioGPT handles clinical protocols before CPT
- **CPT model PPL** on clinical text → how well it handles them after CPT
- **Expected improvement:** 10–40% PPL reduction indicates successful domain adaptation

### 9.2 Catastrophic Forgetting

**What:** When a model trained on new data "forgets" previously learned knowledge.

**Example:** After CPT on clinical text, the model might no longer know that "The capital of France is Paris."

**How it's detected:** Compare base model vs. CPT model outputs on general-knowledge prompts. If CPT outputs are gibberish, repetitive, or factually wrong → forgetting has occurred.

**Mitigation strategies:**
- Use a small learning rate (2e-5 in this notebook)
- Limit training epochs/steps
- Use LoRA/adapters instead of full fine-tuning (used in Assignment 1B)
- Mix general + domain data during training

---

## 10. Inference & Text Generation

### 10.1 Nucleus Sampling (top-p)

At each step, sort tokens by probability. Keep the smallest set of tokens whose cumulative probability ≥ `top_p`. Sample from this set.

```
top_p = 0.95 → keep tokens until cumulative probability reaches 95%
```

**Why:** Adapts to the model's confidence. When the model is certain (one token has 90% probability), the set is small. When uncertain, more tokens are considered.

### 10.2 Top-k Sampling

At each step, keep only the `k` most probable tokens and sample from them.

```
top_k = 50 → only the 50 highest-probability tokens are candidates
```

**Difference from top-p:** Fixed number of candidates regardless of the probability distribution.

### 10.3 Temperature

Scales the logits before softmax, controlling randomness:

$$P(t_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Temperature | Effect |
|-------------|--------|
| T < 1.0 | Sharper distribution → more focused/deterministic |
| T = 1.0 | Original distribution |
| T > 1.0 | Flatter distribution → more random/creative |

**Used:** `temperature=0.7` — slightly more focused than default.

### 10.4 Repetition Penalty

Divides the logit of any token that has already appeared in the generated text by the penalty factor.

```
repetition_penalty = 1.2 → previously seen tokens get their logits ÷ 1.2
```

**Why:** Without this, GPT models often get stuck in loops ("the treatment the treatment the treatment...").

---

## 11. File Formats & Storage

| Format | Used For | Why |
|--------|----------|-----|
| **PDF** | Source clinical documents | Official publication format |
| **TXT** | Extracted/cleaned corpus | Simple, human-readable |
| **Parquet** | Packed tokenized dataset | Columnar, compressed, efficient for large datasets |
| **CSV** | Baseline outputs | Easy to compare in Excel/Pandas |
| **SafeTensors** | Model weights | Safe, fast loading, no pickle vulnerabilities |
| **JSON** | Model config, tokenizer config | Human-readable configuration |

---

## Quick Reference — Hyperparameters Used

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_ID` | `microsoft/biogpt` | Base model |
| `MODEL_MAX_LENGTH` | 1024 | Packed sequence length |
| `MIN_DOC_LENGTH` | 200 chars | Length filter threshold |
| `DEDUP_THRESHOLD` | 0.8 | MinHash similarity cutoff |
| `CPT_EPOCHS` | 3 | Training epochs (GPU) |
| `CPT_BATCH_SIZE` | 2 | Per-device batch size |
| `CPT_LEARNING_RATE` | 2e-5 | Learning rate |
| `CPT_WARMUP_STEPS` | 50 | LR warmup steps |
| `CPT_GRADIENT_ACCUMULATION` | 4 | Gradient accumulation steps |
| `weight_decay` | 0.01 | L2 regularization |
| `lr_scheduler_type` | cosine | LR scheduler |
| `top_k` | 50 | Generation: top-k sampling |
| `top_p` | 0.95 | Generation: nucleus sampling |
| `temperature` | 0.7 | Generation: sampling temperature |
| `repetition_penalty` | 1.2 | Generation: anti-repetition |

---

*This document is for educational/reference purposes only and must not replace professional clinical judgment.*
