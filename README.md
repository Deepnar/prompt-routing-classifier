# Prompt Routing Classifier

A semantic NLP classifier that categorizes user prompts by **topic** and **intent** using sentence embeddings and multi-label classification.

> Built as a **lightweight routing component for multi-model AI systems**, where different prompts are handled by specialized models (coding models, creative models, research models, etc.).

---

## Overview

Modern AI systems often need to **route prompts to the correct subsystem**. For example:

| Prompt | Topic | Intent | Ideal Model |
|--------|-------|--------|-------------|
| "How do I build an API?" | `technical` | `implementation_help` | coding model |
| "Help me plan my career path" | `career` | `strategic_planning` | advisory model |
| "Write a fantasy story" | `creative` | `idea_exploration` | creative model |

Instead of sending every prompt to a single large model, this system first **classifies the prompt**, then routes it intelligently.

This project builds a **semantic prompt classifier** capable of identifying:

- The **topic** of a prompt
- The **intent** behind the prompt

The result is a **multi-label classification system** that serves as the front layer of a prompt-routing architecture.

---

## Project Goals

- Build a **multi-label NLP classifier** for prompt analysis
- Use **semantic embeddings** instead of simple keyword matching
- Create a **lightweight and fast routing component**
- Demonstrate a **complete ML pipeline** from dataset creation to deployment

This repository covers the entire workflow:

- Dataset creation
- Automatic labeling
- Data cleaning
- Embedding generation
- Multi-label classification
- Evaluation
- Inference pipeline

---

## Dataset

The dataset contains approximately **5,100 prompts** extracted from personal chat interactions.

Each prompt is labeled with:

### Topics (7 classes)

| Label | Description |
|-------|-------------|
| `technical` | Programming, engineering, system design |
| `academic` | Research, study, theoretical concepts |
| `career` | Job search, professional growth, planning |
| `project` | Building, shipping, product decisions |
| `creative` | Writing, art, brainstorming |
| `meta_learning` | Learning how to learn, self-improvement |
| `general` | Everything else |

### Intents (9 classes)

| Label | Description |
|-------|-------------|
| `information_request` | Asking for facts or explanations |
| `implementation_help` | Wanting step-by-step guidance |
| `troubleshooting` | Debugging or fixing a problem |
| `decision_making` | Choosing between options |
| `strategic_planning` | Long-term thinking and planning |
| `validation_seeking` | Checking if an approach is correct |
| `idea_exploration` | Brainstorming or creative exploration |
| `constraint_analysis` | Understanding limits or trade-offs |
| `emotional_processing` | Reflecting on feelings or situations |

> Because the dataset contains private conversations, it is **not included in this repository**.

---

## Dataset Creation Pipeline

```
Raw Chat Logs
     ↓
Prompt Extraction
     ↓
LLM-based Auto Labeling
     ↓
Dataset Cleaning
     ↓
Multi-Label Dataset (~5k prompts)
```

An LLM labeling pipeline was used to automatically assign topic and intent labels. The dataset was then cleaned to remove:

- Inconsistent labels
- Malformed entries
- Duplicate prompts

---

## Model Architecture

```
User Prompt
     ↓
SentenceTransformer (all-MiniLM-L6-v2)
     ↓
384-dimensional semantic vector
     ↓
One-vs-Rest Logistic Regression
     ↓
Multi-label classification
     ↓
Topic + Intent prediction
```

### Why this architecture?

| Property | Benefit |
|----------|---------|
| Semantic embeddings | Strong language understanding beyond keywords |
| Logistic Regression | Fast, interpretable, lightweight |
| One-vs-Rest | Supports multi-label output natively |
| No GPU required | Deployable anywhere |

Compared to fine-tuned transformers, this approach is **much cheaper and faster**, making it ideal for real-time routing systems.

---

## Multi-Label Classification

Prompts can carry **multiple topics and intents simultaneously**:

```
Prompt:   "How should I design my startup website?"

Topics:   technical, project
Intent:   strategic_planning
```

To support this, the model uses:

```python
OneVsRestClassifier(LogisticRegression)
```

Each label is treated as an independent binary classifier, allowing any combination of predictions.

---

## Embeddings

Embeddings are generated using:

```
sentence-transformers/all-MiniLM-L6-v2
```

| Property | Value |
|----------|-------|
| Vector dimensions | 384 |
| Inference speed | Very fast |
| Semantic quality | Strong baseline |
| Hardware requirements | CPU only |

Using embeddings instead of TF-IDF significantly improves semantic understanding for short, varied prompts.

---

## Training Process

```
Dataset
  ↓
Train / Test Split (80 / 20)
  ↓
Sentence Embedding Generation
  ↓
MultiLabelBinarizer
  ↓
One-vs-Rest Logistic Regression
  ↓
Model Evaluation
  ↓
Model Serialization (joblib)
```

---

## Performance

Evaluation on a held-out test set (20% of data):

| Task | F1 Score |
|------|----------|
| Topic Classification | **0.68** |
| Intent Classification | **0.56** |

Given the constraints — ~5k prompts, multi-label problem, noisy auto-generated labels — these scores represent a **strong baseline**.

---

## Example Predictions

```
Prompt: "How do I build a REST API?"

Topics:   technical (0.82), project (0.41)
Intents:  implementation_help (0.76), information_request (0.44)
```

```
Prompt: "I want to write a fantasy story"

Topics:   creative (0.91)
Intents:  idea_exploration (0.74)
```

```
Prompt: "I'm not sure if this approach is correct"

Topics:   meta_learning (0.48), technical (0.43)
Intents:  validation_seeking (0.63), decision_making (0.41)
```

---

## Inference Pipeline

```
User Prompt
     ↓
Embedding Generation
     ↓
Probability Prediction
     ↓
Threshold Filtering
     ↓
Fallback Selection (always returns ≥ 1 label)
     ↓
Label Decoding
```

A probability threshold filters predictions while supporting multiple labels per prompt. Fallback logic ensures the system always returns at least one result.

---

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Classifier

```bash
python predict_prompt.py
```

### Example Interaction

```
Enter a prompt: how do i build a website

Topics:   technical, project
Intents:  information_request
```

---

## Project Structure

```
prompt-intent-classifier/
│
├── models/
│   ├── encoder.pkl
│   ├── topic_model.pkl
│   ├── intent_model.pkl
│   ├── topic_binarizer.pkl
│   └── intent_binarizer.pkl
│
├── predict_prompt.py
├── train_prompts.ipynb
├── cleaning_prompts.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Applications

This classifier can be used as a component in:

- **Prompt routing** in multi-model AI systems
- **Conversational AI** intent detection
- **AI tool selection** and orchestration
- **Semantic query categorization**
- **Intelligent assistant** architectures

---

## Limitations

- Relatively small dataset (~5k prompts)
- Noisy labels from automatic annotation
- Intent categories sometimes overlap semantically

These factors primarily affect intent classification accuracy.

---

## Future Improvements

- Expand dataset size with more diverse prompts
- Manual label validation to reduce noise
- Fine-tuned transformer classifier
- Hierarchical classification (topic → intent)
- Full integration with a prompt-routing system

---

## License

This project is released under the [MIT License](LICENSE).
