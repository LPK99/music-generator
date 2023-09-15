import streamlit as st
from audiocraft.models import MusicGen

input_list = []

@st.cache_resource
def load_model(duration):
    duration_ = int(duration)
    model = MusicGen.get_pretrained("melody")
    model.set_generation_params(duration=duration_)
    return model


def generate(model, inputs):
    audio_values = model.generate(inputs)
    sampling_rate = model.sample_rate
    st.audio(audio_values[0].cpu().numpy(), sample_rate=sampling_rate)

def main():
    st.set_page_config(
        page_title="Music Generator",
        page_icon="musical-note.png",
    )
    st.title('Music Generator')
    st.image(image='musical-note.png', width=100)
    prompt_input = st.text_input('Enter your music description')
    input_list.append(prompt_input)
    duration = st.text_input('Enter the duration of the audio')
    if st.button('Create your audio'):
        model = load_model(duration=duration)
        generate(model=model, inputs=input_list)
    st.write("[Checkout github repo](https://github.com/LPK99/music-generator)")
    
if __name__ == '__main__':
    main()