# Assignment 1A — Interview Questions and Answers
## Continual Pre-Training (CPT) for a Clinical Protocol Lookup Assistant

Use these answers as speaking points. They are intentionally concise so they can be expanded during an interview or viva.

---

## A. Project Overview

### 1. What is the objective of this assignment?
The objective is to adapt BioGPT to clinical protocol and guideline text using Continual Pre-Training (CPT), then evaluate whether the adapted model understands the clinical domain better without losing general language ability.

### 2. What is the proposed use case?
The use case is a clinical protocol lookup assistant that can help users retrieve information about treatment protocols, drug dosage guidelines, and clinical pathways. It is an educational/reference system and does not replace professional clinical judgment.

### 3. What is Continual Pre-Training?
CPT is the process of taking an already pretrained language model and training it further on unlabeled, domain-specific text. The objective remains next-token prediction, but the model learns domain vocabulary, style, and recurring patterns.

### 4. How is CPT different from supervised fine-tuning?
CPT uses raw domain text and the causal language modeling objective. Supervised fine-tuning uses labeled instruction-response pairs to teach the model how to follow a task or answer format. CPT improves domain adaptation, while SFT improves instruction following.

### 5. Why was BioGPT selected?
BioGPT is a GPT-style biomedical language model pretrained on PubMed abstracts. It already understands biomedical terminology, so it is a more suitable starting point for clinical adaptation than a general-purpose language model.

### 6. What are the main stages of the pipeline?
The stages are data collection, PDF text extraction, cleaning, deduplication, language filtering, tokenization, sequence packing, model inspection, CPT training, loss analysis, perplexity evaluation, and catastrophic-forgetting evaluation.

---

## B. Data Collection and Cleaning

### 7. What data sources are used?
The corpus uses clinical and public-health guidance from official WHO and NICE sources. Examples include COVID-19 management, HIV antiretroviral therapy, tuberculosis treatment, antimicrobial resistance, essential medicines, and postpartum haemorrhage.

### 8. Why is data cleaning important for CPT?
The model learns directly from the training text, so noisy, duplicated, incomplete, or incorrectly extracted documents can reduce training quality and teach undesirable patterns.

### 9. How is text extracted from PDFs?
PyMuPDF is imported as `fitz`. Each PDF is opened page by page, and plain text is extracted with `page.get_text("text")`. The extracted text is saved as UTF-8 `.txt` files.

### 10. What is a limitation of PDF extraction?
PDF extraction may lose table structure, reading order, headers, and formatting. It also cannot reliably recover text from scanned image-only PDFs without OCR.

### 11. What does the length filter do?
It removes documents shorter than 200 characters after cleaning. Very short extracted files are often cover pages, table-of-contents pages, or incomplete documents.

### 12. What is MinHash?
MinHash creates a compact signature that approximates the Jaccard similarity between two sets of text shingles. It allows similar documents to be compared efficiently without storing every shingle pair.

### 13. What is LSH and why is it used?
Locality-Sensitive Hashing indexes similar MinHash signatures into buckets. It makes near-duplicate lookup much faster than comparing every document with every other document.

### 14. Why are three-word shingles used?
Three-word shingles provide a balance: one-word shingles are too common and cause false matches, while very long shingles are too strict and may miss similar text with small edits.

### 15. What does the deduplication threshold of 0.8 mean?
Documents with an estimated similarity of at least 80% are treated as near duplicates. One copy is retained so repeated boilerplate does not dominate the corpus.

### 16. Why is language filtering applied?
The pipeline keeps English documents because the selected corpus and BioGPT setup are intended for English clinical text. Non-English or multilingual sections could add noise and reduce consistency.

### 17. What is a weakness of detecting language from only the first 1,000 characters?
The first 1,000 characters may not represent the language of the entire document. A stronger pipeline would sample multiple sections or detect language at paragraph level.

---

## C. Tokenization and Packing

### 18. What tokenizer is used?
The tokenizer is loaded from `microsoft/biogpt` using `AutoTokenizer`. BioGPT uses Moses-style preprocessing together with Byte Pair Encoding (BPE).

### 19. What is BPE?
Byte Pair Encoding splits words into frequent subword units. This allows the tokenizer to represent rare medical terms by combining known pieces rather than requiring every complete word to be in the vocabulary.

### 20. Why must the original BioGPT tokenizer be used?
The model's embedding matrix was trained with that vocabulary and token-ID mapping. Using a different tokenizer would make the input IDs incompatible with the pretrained embeddings.

### 21. What are BOS and EOS tokens?
BOS means Beginning of Sequence and EOS means End of Sequence. They mark document boundaries so the model can distinguish where one clinical document ends and another begins.

### 22. Why is EOS also used as the padding token?
GPT-style tokenizers often do not have a dedicated padding token. Reusing EOS avoids adding a new vocabulary entry and is a common practical solution for batching.

### 23. What is sequence packing?
Sequence packing concatenates tokenized documents into one stream and slices that stream into fixed-length chunks, here up to 1,024 tokens. It reduces padding and improves training efficiency.

### 24. What is the trade-off of sequence packing?
A packed chunk can contain the end of one document and the beginning of another. BOS and EOS tokens mark the boundary, but the model can still observe an artificial transition between documents.

### 25. Why is the packed dataset saved as Parquet?
Parquet is a compact, columnar format that supports efficient storage and loading of structured token sequences. It also integrates with Hugging Face Datasets.

### 26. What happens to leftover tokens shorter than the maximum sequence length?
The current packing loop discards the incomplete tail at the end of the token stream. A different implementation could retain it with padding, but that would require an attention mask and careful label handling.

---

## D. Model and Training

### 27. What type of model is BioGPT?
BioGPT is a decoder-only, autoregressive transformer. It predicts the next token using the tokens that appear earlier in the sequence.

### 28. What is causal language modeling?
Causal language modeling trains the model to predict token $t+1$ from tokens up to position $t$. A causal attention mask prevents a position from looking at future tokens.

### 29. What is the purpose of the `lm_head`?
The `lm_head` projects each hidden state to a vector of vocabulary logits. Its output dimension must equal the vocabulary size so the model can assign a score to every possible next token.

### 30. Why does the notebook verify the `lm_head` dimension?
The check confirms that the final projection is compatible with the tokenizer vocabulary. A mismatch would indicate an incompatible or incorrectly loaded model architecture.

### 31. What is cross-entropy loss in this task?
Cross-entropy measures how well the predicted probability distribution matches the actual next token. Lower loss means the model assigns higher probability to the correct target tokens.

### 32. What learning rate is used and why is it small?
The learning rate is `2e-5`. CPT starts from an already useful model, so a small learning rate helps adapt the model gradually and reduces the risk of damaging pretrained knowledge.

### 33. What is learning-rate warmup?
Warmup gradually increases the learning rate from a small value to the target value during the first training steps. This reduces unstable updates at the beginning of training.

### 34. Why is a cosine scheduler used?
A cosine scheduler gradually decreases the learning rate after warmup. This allows larger updates early in training and smaller, more stable updates later.

### 35. What is gradient accumulation?
Gradient accumulation computes gradients over several smaller batches and updates the weights only after they are combined. With batch size 2 and accumulation of 4, the effective batch size is 8, assuming no other distributed workers.

### 36. What is gradient checkpointing?
Gradient checkpointing saves memory by storing fewer intermediate activations during the forward pass and recomputing them during backpropagation. The trade-off is additional computation time.

### 37. What are `bf16` and `fp16` used for?
They are lower-precision numeric formats used on supported GPUs to reduce memory use and improve throughput. The notebook prefers BF16 when supported and otherwise uses FP16; CPU execution falls back to FP32.

### 38. Why is weight decay included?
Weight decay acts as L2-style regularization. It discourages excessively large parameter updates and can reduce overfitting to a small clinical corpus.

### 39. Why is a CPU-specific training path included?
Training a 347-million-parameter model on CPU is slow, so the notebook uses fewer steps, a smaller batch, and lower accumulation for a sanity check. Full CPT is intended for a GPU.

### 40. What is the purpose of the custom loss callback?
The `LossLoggerCallback` stores loss values and their training steps whenever the Hugging Face Trainer logs them. These values are later used to plot and interpret the training curve.

---

## E. Evaluation

### 41. What is perplexity?
Perplexity is the exponential of the average causal language-model loss. Lower perplexity means the model is less surprised by the evaluation text and predicts it more confidently.

$$\mathrm{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_i\right)$$

### 42. How is domain adaptation measured?
The notebook compares the base model's perplexity and the CPT model's perplexity on held-out clinical documents. A lower CPT perplexity indicates improved modeling of the clinical domain.

### 43. Why must the evaluation documents be held out?
Evaluating on unseen documents gives a better estimate of generalization. Measuring only training documents could make the model appear successful because it has memorized them.

### 44. What does a PPL reduction indicate?
A PPL reduction suggests that CPT improved the model's ability to predict clinical language. It does not by itself prove factual correctness, safe dosage recommendations, or clinical usefulness.

### 45. What is catastrophic forgetting?
Catastrophic forgetting occurs when adaptation to a new domain causes the model to lose previously learned general knowledge or language ability.

### 46. How is catastrophic forgetting checked here?
The notebook generates text from general prompts using both the original base model and the CPT model, then compares the outputs. Very short, repetitive, or nonsensical CPT outputs may indicate degradation.

### 47. Is the current forgetting test sufficient?
No. It is a qualitative and very small test. A stronger evaluation would use a larger fixed benchmark, deterministic decoding, factual accuracy checks, and quantitative metrics such as general-domain perplexity.

### 48. Why should generation use deterministic settings for a fair comparison?
Sampling introduces randomness, so two outputs may differ even when the models are equally capable. Greedy decoding or fixed-seed sampling makes the base-versus-CPT comparison more reproducible.

### 49. What does a decreasing loss curve mean?
It usually means the model is learning to predict the training data better. However, training loss alone cannot confirm that the model generalizes or gives medically correct answers.

### 50. What would you do if domain perplexity increased after CPT?
I would check the data and tokenization first, then reduce the learning rate, increase warmup, check the evaluation split, reduce the number of epochs, and confirm that the checkpoint was saved and loaded correctly.

---

## F. Critical Thinking and Limitations

### 51. Can this model be used to prescribe treatment?
No. It is an educational/reference prototype. Clinical decisions require current validated guidelines, patient-specific information, qualified professionals, and appropriate safety checks.

### 52. Does lower perplexity guarantee factual correctness?
No. Perplexity measures token prediction, not truth. A model can confidently generate an incorrect or outdated clinical statement.

### 53. What are the main risks of this clinical application?
Risks include outdated guidelines, hallucinated dosages, incorrect contraindications, source extraction errors, biased coverage, and users treating generated text as medical advice.

### 54. How could the assistant be made safer?
Use retrieval from versioned official sources, show citations and publication dates, add a clinical validation layer, constrain dosage outputs, require clinician review, log responses, and clearly display uncertainty and the medical disclaimer.

### 55. Why might CPT alone not be enough for a question-answering assistant?
CPT teaches the model domain language but does not guarantee reliable instruction following or grounded retrieval. A production assistant would usually combine a domain-adapted model with retrieval-augmented generation and instruction tuning.

### 56. What is one limitation in the current train/evaluation design?
The notebook builds the packed dataset from the full cleaned corpus and later selects a held-out evaluation split from that same corpus. For a strict unseen evaluation, the documents should be split before tokenization and only the training portion should be packed for CPT.

### 57. Why should the base and CPT models be compared using the same evaluation data?
Using the same prompts and documents controls the experiment. Differences in output or perplexity are then more likely to be caused by CPT rather than by different evaluation samples.

### 58. What would you improve first in this notebook?
I would create a document-level train/validation/test split before packing, use a fixed evaluation benchmark, make generation deterministic, report confidence intervals where possible, and add citation-based factual evaluation.

### 59. What is the difference between memorization and domain learning?
Memorization is reproducing training text, while domain learning is acquiring reusable vocabulary, structure, and knowledge patterns that generalize to unseen clinical documents. Held-out evaluation helps distinguish them, although it does not eliminate the problem completely.

### 60. Summarize the project in one minute.
I started with BioGPT, a biomedical decoder-only language model, and continued training it on cleaned WHO and NICE clinical guidance. The pipeline extracts PDF text, removes short and duplicate documents, keeps English content, tokenizes with the original BioGPT tokenizer, packs sequences, and trains with causal language modeling. I evaluate adaptation using domain perplexity and check general-language retention to detect catastrophic forgetting. The result is an educational clinical lookup prototype, not a replacement for professional medical advice.

---

## Quick Formula and Parameter Reference

| Item | Notebook value or meaning |
|---|---|
| Base model | `microsoft/biogpt` |
| Approximate parameters | 347 million |
| Maximum sequence length | 1,024 tokens |
| Minimum document length | 200 characters |
| MinHash similarity threshold | 0.8 |
| MinHash permutations | 128 |
| CPT epochs on GPU | 3 |
| CPT learning rate | `2e-5` |
| GPU batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Warmup steps | 50 |
| Objective | Causal next-token prediction |
| Main domain metric | Perplexity |
| Main retention check | General-prompt comparison |

## Final Interview Reminder

Do not claim that the model is clinically accurate merely because its loss or perplexity decreases. State clearly that the experiment demonstrates domain adaptation and language-model evaluation, while factual validation, retrieval grounding, source citation, and clinical governance are still required for real deployment.

## Built-in and Library Functions Reference

Strictly speaking, functions such as `len()` and `max()` are Python built-ins. Functions such as `torch.no_grad()` and `AutoTokenizer.from_pretrained()` come from external libraries. The following table includes both types because all of them are used in the notebook.

| Function or method | Source | Purpose in this project |
|---|---|---|
| `print()` | Python built-in | Display device information, training progress, metrics, and reports. |
| `len()` | Python built-in | Count documents, tokens, dataset rows, or model layers. |
| `sum()` | Python built-in | Calculate total characters or total model parameters. |
| `max()` | Python built-in | Select the document-removal step with the greatest impact and protect the evaluation split from being zero. |
| `range()` | Python built-in | Iterate through pages, token chunks, and training or evaluation items. |
| `enumerate()` | Python built-in | Track prompt and document numbers while iterating. |
| `zip()` | Python built-in | Pair questions with their completion prompts. |
| `open()` | Python built-in | Read and write extracted, cleaned, and saved text files. |
| `os.path.join()` | `os` | Build platform-independent file paths. |
| `os.path.exists()` | `os` | Check whether downloaded PDFs or model checkpoints already exist. |
| `os.makedirs()` | `os` | Create corpus and output directories. |
| `glob.glob()` | `glob` | Find PDF and text files using filename patterns. |
| `re.sub()` | `re` | Remove unwanted characters and normalize whitespace during cleaning. |
| `Path.stem` | `pathlib` | Get a PDF filename without its extension when creating a text filename. |
| `requests.get()` | `requests` | Download clinical PDFs from official sources. |
| `fitz.open()` | PyMuPDF | Open PDFs for page-by-page text extraction. |
| `page.get_text()` | PyMuPDF | Extract plain text from each PDF page. |
| `detect()` | `langdetect` | Detect the language of a document sample. |
| `MinHash()` | `datasketch` | Create a compact similarity signature for a document. |
| `MinHash.update()` | `datasketch` | Add each text shingle to a MinHash signature. |
| `MinHashLSH()` | `datasketch` | Create an index for efficient near-duplicate detection. |
| `lsh.query()` | `datasketch` | Find documents with similar MinHash signatures. |
| `lsh.insert()` | `datasketch` | Add a unique document signature to the similarity index. |
| `AutoTokenizer.from_pretrained()` | Transformers | Load the tokenizer associated with BioGPT. |
| `tokenizer.encode()` | Transformers | Convert clinical text into token IDs. |
| `tokenizer()` | Transformers | Tokenize prompts or evaluation text into model input tensors. |
| `tokenizer.decode()` | Transformers | Convert generated token IDs back into readable text. |
| `tokenizer.save_pretrained()` | Transformers | Save the tokenizer with the CPT checkpoint. |
| `AutoModelForCausalLM.from_pretrained()` | Transformers | Load the base BioGPT model or the trained CPT model. |
| `model.generate()` | Transformers/PyTorch | Generate text continuations from clinical or general prompts. |
| `torch.device()` | PyTorch | Select CPU or CUDA as the execution device. |
| `torch.cuda.is_available()` | PyTorch | Check whether a CUDA-compatible GPU is available. |
| `torch.no_grad()` | PyTorch | Disable gradients during inference and perplexity evaluation to save memory. |
| `torch.tensor()` | PyTorch | Convert token lists into tensors for model training. |
| `torch.cuda.empty_cache()` | PyTorch | Release unused cached GPU memory between model evaluations. |
| `Dataset.__getitem__()` | PyTorch | Return one packed training example containing inputs, masks, and labels. |
| `HFDataset.from_dict()` | Hugging Face Datasets | Create a dataset from packed token sequences. |
| `packed_dataset.to_parquet()` | Hugging Face Datasets | Save the packed dataset in Parquet format. |
| `load_dataset()` | Hugging Face Datasets | Load the saved Parquet dataset for training. |
| `Trainer()` | Transformers | Manage the CPT training loop, optimization, logging, and checkpointing. |
| `trainer.train()` | Transformers | Start model training on the packed clinical dataset. |
| `DataCollatorForLanguageModeling()` | Transformers | Prepare causal-language-model batches with `mlm=False`. |
| `model.eval()` | PyTorch | Switch the model to evaluation mode before measuring perplexity. |
| `pandas.DataFrame()` | pandas | Organize baseline, comparison, and forgetting results into tables. |
| `DataFrame.to_csv()` | pandas | Save evaluation results and baseline outputs as CSV files. |
| `np.random.shuffle()` | NumPy | Randomize document order before selecting the evaluation split. |
| `np.exp()` | NumPy | Convert average cross-entropy loss into perplexity. |
| `np.mean()` | NumPy | Calculate average document token length. |
| `plt.plot()` | Matplotlib | Plot the CPT training loss curve. |
| `plt.savefig()` | Matplotlib | Save the loss plot to the output directory. |
| `tqdm()` | tqdm | Display progress bars during extraction, tokenization, and evaluation. |

### Common Viva Question: What is the difference between a function and a method?

A function can be called independently, such as `len(text)` or `print(value)`. A method belongs to an object or class and is called with dot notation, such as `tokenizer.encode(text)`, `model.eval()`, or `dataframe.to_csv(path)`.
