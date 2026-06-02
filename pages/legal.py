import streamlit as st
import torch
import json
from model import GPT 

vocab_size = 195

@st.cache_resource
def load_assets():
    with open("assets/char_mapping_legal.json", "r") as f:
        data = json.load(f)
    
    atoi = data["atoi"]
    itoa = data["itoa"]
    vocab_size = len(atoi)

    model = GPT(vocab_size=vocab_size, d_model=512, block_size=256, n_layer=8, n_head=8, dropout=0.1)
    
    state_dict = torch.load("assets/legal_gpt_weights.pth", map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, atoi, itoa

model, atoi, itoa = load_assets()

st.title("Legal Transformer")
st.markdown("Generates stylized contract boilerplate based on the Atticus CUAD dataset.")

prompt = st.text_input("Enter a prompt:", "Section 1. Definitions.")

st.sidebar.header("Generation Parameters")

max_tokens = st.sidebar.slider(
    "Max Tokens to Generate", 
    min_value=50, 
    max_value=500, 
    value=200, 
    step=10
)

temperature = st.sidebar.slider(
    "Temperature (Creativity)", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.7, 
    step=0.1
)

top_k = st.sidebar.slider(
    "Top-K Sampling", 
    min_value=1, 
    max_value=50, 
    value=3
)

if st.button("Generate Clause"):
    context_tokens = [atoi.get(ch, 0) for ch in prompt]
    input_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(
            input_tensor, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
        output_text = ''.join([itoa[str(idx)] for idx in generated[0].tolist()])
    st.write(output_text)