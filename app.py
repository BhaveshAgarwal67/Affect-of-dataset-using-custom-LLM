import streamlit as st

st.set_page_config(page_title="LLM Comparison Lab", layout="wide")

st.title("Generative AI: Architecture Comparison")

st.markdown("""
Welcome to my Transformer Research Dashboard. 

This application demonstrates the impact of training data on the output of character-level GPT models built from scratch using PyTorch. 

### Project Overview
Using inspiration from Google's paper "Attention is All You Need", I implemented a character-level GPT architecture with multi-head self-attention and feedforward layers. There are total 8 transformer blocks, with model dimension of 512. There are total of approx. 25 million parameters in the model.

By training the Transformer architectures on vastly different datasets, we can observe how the "statistical diet" of a model influences its generative behavior:

* **Legal Transformer**: Trained on formal contract boilerplate (Atticus CUAD dataset). This model emphasizes structural precision and rigid syntax.
* **Fiction Narrative**: Trained on narrative fiction (Harry Potter dataset). This model demonstrates higher variance and fluid storytelling capabilities.

---

### How to use the models
Use the **sidebar** to navigate between the models. Each page allows you to:
1. Input a custom prompt.
2. Adjust generation parameters (max tokens, temperature, top-k sampling).
3. Generate text using the model's specialized weights.
4. Compare the structural differences in the output.

---

## Limitations
Since the models are character-level and trained on relatively small datasets, they may produce incoherent or repetitive text. The generated output is more of a stylistic demonstration rather than a fully functional language model.

---

### Engineering Notes
This project was containerized using **Docker** for environment parity and utilizes **modular architecture** to ensure clean, maintainable code.
""")

st.info("Check out the source code and training methodology at [GitHub](https://github.com/BhaveshAgarwal67/Affect-of-dataset-using-custom-LLM)")