import streamlit as st
import torchaudio
import io
import matplotlib.pyplot as plt
# Assuming audio_recorder_streamlit is a custom or third-party module for recording audio in Streamlit apps
from audio_recorder_streamlit import audio_recorder
from trainer import SpeechLLMLightning
import re 
import json
import sys

def load_model(ckpt_path):
    model = SpeechLLMLightning.load_from_checkpoint(ckpt_path)
    tokenizer = model.llm_tokenizer
    model.eval()
    model.freeze()
    model.to('cuda')
    return model, tokenizer

def get_or_load_model(ckpt_path):
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        model = SpeechLLMLightning.load_from_checkpoint(ckpt_path)
        tokenizer = model.llm_tokenizer
        model.eval()
        model.freeze()
        model.to('cuda')
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
    return st.session_state.model, st.session_state.tokenizer

def extract_dictionary(input_string):
    # Extract the JSON-like string
    json_str_match = re.search(r'\{.*\}', input_string)
    if not json_str_match:
        print(input_string)
        return "No valid JSON found."
    
    json_str = json_str_match.group(0)
    
    # Attempt to fix common JSON formatting issues:
    # 1. Ensure property names are enclosed in double quotes.
    # 2. Remove trailing commas before closing braces or brackets.
    json_str = re.sub(r'(?<=\{|\,)\s*([^\"{}\[\]\s]+)\s*:', r'"\1":', json_str)  # Fix unquoted keys
    json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)  # Remove trailing commas
    
    try:
        # Parse the corrected JSON string into a dictionary
        data_dict = json.loads(json_str)
        return data_dict
    except json.JSONDecodeError as e:
        # Return an error message if JSON parsing fails
        return f"Error parsing JSON: {str(e)}"

# Function to generate a response from the model
def generate_response(wav, model, tokenizer):
    pre_speech_prompt = '''Instruction:
    Give me the following information about the audio [SpeechActivity, Transcript, Gender, Age, Emotion, Accent]

    Input: 
    <speech>'''

    post_speech_prompt = '''</speech> 

    Output:'''

    output_prompt = '\n<s>'

    pre_tokenized_ids = tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
    post_tokenized_ids = tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
    output_tokenized_ids = tokenizer(output_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

    combined_embeds, atts, label_ids = model.encode(wav.cuda(), pre_tokenized_ids.cuda(), post_tokenized_ids.cuda(), output_tokenized_ids.cuda())
    out = model.llm_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=2000,
        ).cpu().tolist()[0]

    output_text = tokenizer.decode(out, skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    model, tokenizer = get_or_load_model("path-to-best_checkpoint.ckpt")

    # Streamlit UI components
    st.title("SpeechLLM : Multi-Modal LLM for Speech Understanding")
    
    st.markdown("""
    [![hf_model](https://img.shields.io/badge/🤗-SpeechLLM%20HuggingFace-blue.svg)](https://huggingface.co/collections/skit-ai/speechllm-66605bfb37a54d4e4a60efe2)
    [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/skit-ai/SpeechLLM/blob/main/LICENSE)
    [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/skit-ai/SpeechLLM.git)
    [![GitHub stars](https://img.shields.io/github/stars/skit-ai/SpeechLLM?style=social)](https://github.com/skit-ai/SpeechLLM/stargazers)
    [![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?logo=googlecolab&color=blue)](https://colab.research.google.com/drive/1uqhRl36LJKA4IxnrhplLMv0wQ_f3OuBM?usp=sharing)
    """, unsafe_allow_html=True)

    st.write("Click below to record an audio file to get its transcription and other metadata.")

    # Improved layout for audio recording button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("###")
        st.write("###")
        audio_data = audio_recorder(sample_rate=16000, text="")
        # st.write("Click to record")

    # Transcription process
    if audio_data is not None:
        with st.spinner('Transcribing...'):
            try:
                # Load audio data into a tensor
                audio_buffer = io.BytesIO(audio_data)
                st.audio(audio_data, format='audio/wav', start_time=0)
                wav_tensor, sample_rate = torchaudio.load(audio_buffer)
                wav_tensor = wav_tensor.to('cuda').mean(0).unsqueeze(0) # mean of dual channel, remove if audio is mono
                
                # Process audio to get transcription
                transcription = generate_response(wav_tensor.cuda(), model, tokenizer)
                
                # Display the transcription
                st.success('Transcription Complete')
                st.text_area("LLM Output:", value=extract_dictionary(transcription), height=200, max_chars=None)
                # st.code(extract_dictionary(transcription), language='python')
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
