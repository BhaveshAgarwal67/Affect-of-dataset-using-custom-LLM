# Transformer Engine: Custom Character-Level LLM

A high-performance, custom-built Transformer architecture trained from scratch on specialized datasets. This project explores the deep implementation of self-attention mechanisms and generative inference pipelines, bridging the gap between raw tensor manipulation and a production-grade, containerized web application.

## Useful Links

[Live Demo](https://huggingface.co/spaces/BhaveshAgarwal67/Mini_LLM)

[Kaggle Notebook](https://www.kaggle.com/code/bhaveshagarwal67/harry-potter-llm)

---

## Project Structure

Understanding the architecture is key. Here is how the components are organized for modularity and maintainability:

```text
root/
├── app.py                # Central Dashboard & Hub
├── model.py              # Shared Transformer Architecture
├── pages/
│   ├── Legal.py        # Legal-specific inference page
│   └── Potter.py      # Harry-Potter-specific inference page
├── assets/               # Trained weights (.pth) & mappings (.json)
├── Dockerfile            # Dockerfile
└── requirements.txt      # Project dependencies

```

---

## Model Specifications

The model utilizes a standard GPT-style Transformer architecture. Below are the structural parameters used across the trained models:

| Component | Specification |
| --- | --- |
| **Embedding Dimension ($d_{model}$)** | 512 |
| **Transformer Layers ($n_{layer}$)** | 8 |
| **Attention Heads ($n_{head}$)** | 8 |
| **Context Window ($block\_size$)** | 256 |
| **Dropout Rate** | 0.1 |
| **Total Parameter Count** | ~25.4M |

---

## Engineering Core

This project is an end-to-end implementation of an autoregressive language model, focusing on the technical challenges of deep learning architecture.

### Architecture Highlights

* **From-Scratch Implementation**: Built the core Transformer components (Multi-Head Attention, Feed-Forward Networks, LayerNorm) using PyTorch — no pre-trained weights or external LLM APIs were used.
* **Modular Codebase**: Wrote a shared `model.py` that decouples architecture from inference, allowing the app to swap between entirely different trained models (e.g., Legal boilerplate vs. Harry Potter) seamlessly.
* **Production Pipeline**: Utilized **Docker** for environment parity and **Streamlit** to provide an interactive, responsive inference dashboard.

### Key Technical Challenges Resolved

* **Weight Serialization**: Developed a custom loader to handle `DataParallel` artifacts, stripping `module.` prefixes from serialized weights to enable portability between training environments and single-CPU inference.
* **Inference Tuning**: Implemented a professional generation interface featuring configurable `temperature` and `top_k` sampling to mitigate the "stutter/looping" behaviors common in character-level LLMs.
* **Efficient Memory Management**: Utilized `st.cache_resource` for intelligent model loading, ensuring the application remains snappy without redundant reloads of heavy neural network weights.

---

## Tech Stack & Requirements

This project requires `python 3.12+`. Install dependencies via:

```bash
pip install -r requirements.txt

```

**Key Dependencies:**

* `torch`: For the neural network architecture.
* `streamlit`: For the interactive web interface.

---

## Experiments

* **Legal Domain**: Trained on the *Atticus CUAD* dataset to study the model's ability to replicate rigid syntactic structures and legal "boilerplates."
* **Fiction Domain**: Trained on Harry Potter books to evaluate the model’s capacity for maintaining character naming and flow in long-form generation.

---

## Comparative Analysis: Dataset Influence
By training the identical Transformer architecture on disparate datasets, we can observe significant divergence in the emergent properties of the models:

| Feature | Legal Transformer (Atticus CUAD) | Harry Potter Books |
| :--- | :--- | :--- |
| **Syntactic Pattern** | Highly rigid, repetitive clause structures. | Fluid, varied, and sentence-length diverse. |
| **Vocabulary Usage** | High density of technical/contractual jargon. | High density of descriptive/emotive language. |
| **Context Retention** | Excels at maintaining logical consistency. | Excels at maintaining character and plot flow. |
| **Primary Utility** | Boilerplate generation and structural analysis. | Creative writing and narrative exploration. |

The differences observed highlight how the **statistical distribution of the training data** forces the weights within the same architecture to specialize—effectively creating "domain-specific" neural biases that dictate the model's creative or formal output.

---

## How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/BhaveshAgarwal67/Affect-of-dataset-using-custom-LLM.git
cd Affect-of-dataset-using-custom-LLM
```

2. **Build the containerized environment:**
```bash
docker build -t transformer .
```


3. **Launch the application:**
```bash
docker run -p 8501:8501 transformer
```


*Access the app at `http://localhost:8501`*

---

## Ethical Usage

This model is a research project designed to demonstrate the mechanics of neural networks. It generates synthetic text based on the statistical distribution of its training data and does not possess genuine logical reasoning or domain-specific expertise.

---

## Credits & Acknowledgments

* **Core Architecture Inspired by:** 'Attention is all you need' Paper from Google
* **Datasets:** Atticus CUAD (Legal) and Harry Potter Books.

