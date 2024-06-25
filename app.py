import gradio as gr
import openai
import anthropic
import google.generativeai as genai
from transformers import pipeline
import requests
from pydub import AudioSegment
from io import BytesIO
import tempfile

# Initialize the models
model_gpt2 = pipeline("text-generation", model="gpt2")
model_opt = pipeline("text-generation", model="facebook/opt-350m")

def query_model(model_choice, user_question, max_length, preprompt, api_key=None):
    full_prompt = preprompt + " " + user_question if preprompt else user_question
    
    if model_choice == "GPT-2":
        response = model_gpt2(full_prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text']
    elif model_choice == "OPT-350M":
        response = model_opt(full_prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text']
    elif model_choice == "OpenAI":
        if not api_key:
            return "API key is required for OpenAI."
        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return completion.choices[0].message.content
    elif model_choice == "Anthropic":
        if not api_key:
            return "API key is required for Anthropic."
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-2.1",
            max_tokens=max_length,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return message.content[0].text
    elif model_choice == "Gemini":
        if not api_key:
            return "API key is required for Gemini."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(full_prompt)
        return response.text
    else:
        return "Invalid model selected."

def text_to_speech(text, api_key, voice_id, model_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Error in text-to-speech API: {response.status_code} - {response.text}")

def save_audio_file(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio.flush()
        return temp_audio.name

def process_input(model_choice, user_question, max_length, preprompt, api_key, use_tts, tts_api_key, tts_voice_id, tts_model_id, history):
    response = query_model(model_choice, user_question, max_length, preprompt, api_key)
    history = history + [(user_question, response)]
    
    audio_output = None
    if use_tts:
        try:
            audio_data = text_to_speech(response, tts_api_key, tts_voice_id, tts_model_id)
            audio_file = save_audio_file(audio_data)
            audio_output = gr.Audio(value=audio_file, autoplay=True)
        except Exception as e:
            print(f"TTS Error: {str(e)}")
    
    return history, "", audio_output, gr.update(visible=use_tts)

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Model Text Generation with Text-to-Speech")
    
    with gr.Row():
        with gr.Column(scale=2):
            model_choice = gr.Radio(
                choices=["GPT-2", "OPT-350M", "OpenAI", "Anthropic", "Gemini"],
                label="Select Model",
                interactive=True
            )
            api_key = gr.Textbox(
                label="API Key (required for API models)",
                type="password",
                lines=1,
                placeholder="Enter your API key here...",
                visible=False
            )
            api_key_info = gr.Markdown(visible=False)
            max_length = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length")
            preprompt = gr.Textbox(label="Preprompt (optional)", lines=2, placeholder="Enter a preprompt here...")
            user_question = gr.Textbox(label="Enter your question", lines=2, placeholder="Type your question here...")
        
        with gr.Column(scale=1):
            use_tts = gr.Checkbox(label="Use Text-to-Speech", value=False)
            tts_api_key = gr.Textbox(
                label="ElevenLabs API Key",
                type="password",
                lines=1,
                placeholder="Enter your ElevenLabs API key here...",
                visible=False
            )
            tts_voice_id = gr.Textbox(
                label="ElevenLabs Voice ID",
                lines=1,
                placeholder="Enter your ElevenLabs Voice ID here...",
                visible=False
            )
            tts_model_id = gr.Textbox(
                label="ElevenLabs Model ID",
                lines=1,
                placeholder="Enter your ElevenLabs Model ID here...",
                visible=False
            )
    
    query_button = gr.Button("Submit")
    conversation_history = gr.Chatbot(label="Conversation History")
    audio_output = gr.Audio(label="Text-to-Speech Output", visible=False)
    
    def update_api_key_visibility(model):
        api_key_visible = model in ["OpenAI", "Anthropic", "Gemini"]
        api_key_info_md = ""
        if api_key_visible:
            if model == "OpenAI":
                api_key_info_md = "Get your OpenAI API key [here](https://platform.openai.com/api-keys)"
            elif model == "Anthropic":
                api_key_info_md = "Get your Anthropic API key [here](https://console.anthropic.com/settings/keys)"
            elif model == "Gemini":
                api_key_info_md = "Get your Gemini API key [here](https://aistudio.google.com/app/apikey?hl=ja)"
        
        return gr.update(visible=api_key_visible), gr.update(visible=api_key_visible, value=api_key_info_md)
    
    model_choice.change(
        fn=update_api_key_visibility,
        inputs=[model_choice],
        outputs=[api_key, api_key_info]
    )
    
    def update_tts_visibility(use_tts):
        return [gr.update(visible=use_tts) for _ in range(3)]
    
    use_tts.change(
        fn=update_tts_visibility,
        inputs=[use_tts],
        outputs=[tts_api_key, tts_voice_id, tts_model_id]
    )
    
    query_button.click(
        fn=process_input,
        inputs=[model_choice, user_question, max_length, preprompt, api_key, use_tts, tts_api_key, tts_voice_id, tts_model_id, conversation_history],
        outputs=[conversation_history, user_question, audio_output, audio_output]
    )

demo.launch()