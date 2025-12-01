
# Project Part Two – Emotionally Aligned Conversational Generation  
**WASSA 2024 – Track 2 (Emotion, Empathy & Polarity Conditioned Dialogue Generation)**

This repository contains all deliverables required for **Project Part Two** including:
- Q1: Corpus-Based Retrieval Chatbot  
- Q2: In-Context Learning (ICL) LLM Chatbot  
- Q3: Qualitative Evaluation Report  
- All required CSV outputs and datasets  

The project follows the official WASSA Track‑2 task setup.

---

## 📁 **Contents of This Submission**

```
NLP_P2.ipynb                   # Main notebook containing Q1 + Q2 implementations
generations_corpus.csv         # Q1: Corpus-based generations for test set
generations_icl.csv            # Q2: ICL-based generations for test set
trac2_CONVT_train.csv          # Training split
trac2_CONVT_dev.csv            # Development split
trac2_CONVT_test.csv           # Test split
goldstandard_CONVT_test.csv    # Gold test labels (not used for generation)
NLP-P2_Report.pdf              # Q3 qualitative evaluation report
README.md                      # This file
```

---

# 1. Q1 – Corpus-Based Retrieval Chatbot

### **Overview**
The Q1 system generates future conversation turns using **nearest-neighbor retrieval** from the training corpus.  
A weighted similarity score is computed using:

- SentenceTransformer embeddings (`all-mpnet-base-v2`)
- Emotion similarity (scaled to [0,1])
- Empathy similarity (scaled to [0,1])
- EmotionalPolarity one-hot match

### **Similarity Score**

```
S_total = 0.60 * text_similarity
        + 0.15 * emotion_similarity
        + 0.15 * empathy_similarity
        + 0.10 * polarity_match
```

### **Process**
1. Build contextual history using turns 1–5  
2. Compute embedding + label-based similarity with **every** training utterance  
3. Select highest‑scoring utterance as the generated turn  
4. Generate 5 dev turns and 10 test turns  

### **Output**
`generations_corpus.csv`

---

# 2. Q2 – In-Context Learning (ICL) LLM Chatbot

### **Model Used**
```
microsoft/Phi-3-mini-4k-instruct
```
(Selected due to Colab GPU constraints)

### **Few-Shot Prompt Design**
Each example includes:
```
Example:
Context:
Turn 1 ...
Turn 5 ...

Target emotion intensity: X
Target empathy: Y
Target polarity: Z

Expected future responses:
Turn 6 ...
Turn 15 ...
```

### **Pipeline**
1. Randomly sample 3 conversations from training split  
2. Construct few-shot prompt  
3. Provide dev/test context (turns 1–5)  
4. Include emotional targets from turn 6  
5. Generate turns 6–15 using the LLM  
6. Clean/parse model output  

### **Hyperparameters**
```
max_new_tokens = 256
temperature = 0.7
top_p = 0.9
few_shot_examples = 3
```

### **Output**
`generations_icl.csv`

---

# 3. Development Set Results

## **Q1 – Corpus-Based**

| Metric | Score |
|--------|--------|
| ROUGE-1 | ~0.21 |
| ROUGE-2 | ~0.06 |
| ROUGE-L | ~0.18 |
| BLEU | ~0.07 |
| BERTScore F1 | ~0.78 |

Notes:
- Fluent but generic  
- Emotionally flat  
- Heavily repetitive  

---

## **Q2 – In-Context LLM (40-dev subset)**

| Few-Shot Setting | BLEU-like |
|------------------|------------|
| 3-shot | ~0.15 |
| 5-shot | ~0.17 |

Notes:
- More coherent  
- Stronger emotional alignment  
- Minor prompt leakage (“Example:”)  

---

# 4. Preprocessing & Model Choices

### **Preprocessing**
- Normalized `Emotion`, `Empathy` → [0,1]  
- One-hot encoded `EmotionalPolarity`  
- Removed empty text rows  
- Context built via simple concatenation  

### **Why MPNet for embeddings?**
- High semantic matching quality  
- Well-suited for conversational similarity  

### **Why Phi-3 ICL?**
- Lightweight & performant  
- Handles long multi-turn prompts well  

---

# 5. Instructions to Run

## **Recommended: Google Colab**

### Step 1 — Upload datasets into `/content`
```
trac2_CONVT_train.csv
trac2_CONVT_dev.csv
trac2_CONVT_test.csv
```

### Step 2 — Install dependencies
```python
!pip install sentence-transformers evaluate transformers accelerate
```

### Step 3 — Run Q1
Open & execute:
```
NLP_P2.ipynb  (Q1 section)
```

### Step 4 — Run Q2
Execute:
```
NLP_P2.ipynb  (Q2 section)
```

### Step 5 — Outputs generated:
```
generations_corpus.csv
generations_icl.csv
```

---

# 6. Notes & Limitations

### Corpus-Based Model
- Strengths: Simple, deterministic  
- Weaknesses: Repetitive, low emotional expression  

### ICL Model
- Strengths: High emotional alignment, coherent  
- Weaknesses: Occasional prompt leakage, GPU required  

---

# 7. Q3 – Qualitative Evaluation Report

A detailed evaluation of 5 dev conversations is included in:
```
NLP-P2_Report.pdf
```
This includes:
- Part‑1 classifier emotion predictions  
- Ratings for: Fluency, Relevance, Coherence, Emotional Alignment  
- Conversation-level analysis  
- Final model comparison  

---

# 8. Authors & Credits

This project was completed as part of  
**WASSA 2024 Track‑2 – Emotionally Grounded Dialogue Generation**.

