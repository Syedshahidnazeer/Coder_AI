import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load CodeBERT and InCoder models and tokenizers
codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModelForCausalLM.from_pretrained("microsoft/codebert-base")

incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

# Title of the app
st.title("Mini LLMs with Streamlit")

# User input text
input_text = st.text_area("Enter your code or text:")

# Select the model to use
model_choice = st.selectbox("Choose a model:", ["CodeBERT", "InCoder"])

# Generate output
if st.button("Generate"):
    if model_choice == "CodeBERT":
        inputs = codebert_tokenizer(input_text, return_tensors="pt")
        outputs = codebert_model.generate(**inputs)
        result = codebert_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        inputs = incoder_tokenizer(input_text, return_tensors="pt")
        outputs = incoder_model.generate(**inputs)
        result = incoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write("Generated Output:")
    st.write(result)
