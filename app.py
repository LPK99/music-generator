import streamlit as st
from audiocraft.models import MusicGen
import torch
import gc

input_list = []

from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr

def improve(audio):
  

  sr = 44100


  reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

  board = Pedalboard([
    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-16, ratio=2.5),
    LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
    Gain(gain_db=10)
  ])

  effected = board(reduced_noise, sr)
  
  return effected




def clear_cuda_memory():
    torch.cuda.empty_cache()
    st.cache_resource.clear()
    torch.cuda.empty_cache()
    gc.collect()

@st.cache_resource()
def load_model(duration):
    duration_ = int(duration)
    model = MusicGen.get_pretrained("melody")
    model.set_generation_params(duration=duration_)
    return model


def generate(model, inputs):
    audio_values = improve(model.generate(inputs)[0].cpu().numpy())
    print(audio_values)
    sampling_rate = model.sample_rate
    st.audio(audio_values, sample_rate=sampling_rate)

def main():
    st.set_page_config(
        page_title="Music Generator",
        page_icon="musical-note.png",
    )
    cache = torch.cuda.memory_cached() / 1024 ** 3
    print(cache)
    if cache >= 4.6 :
        clear_cuda_memory()
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1024 ** 3:.2f} GB")
    st.title('Music Generator')
    st.image(image='musical-note.png', width=100)
    prompt_input = st.text_input('Enter your music description')
    input_list.append(prompt_input)
    duration = st.text_input('Enter the duration of the audio')
    if st.button('Create your audio'):
        with st.spinner("Your audio is being generated"):
            model = load_model(duration=duration)
            generate(model=model, inputs=input_list)
    st.write("[Check out github repo](https://github.com/LPK99/music-generator)")
    
if __name__ == '__main__':
    main()