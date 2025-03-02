import gradio as gr
import os
import dashscope
import sys
from typing import Generator, Optional, Tuple

sys.path.insert(1, "../cosyvoice")
sys.path.insert(1, "../sensevoice")
sys.path.insert(1, "../cosyvoice/third_party/AcademiCodec")
sys.path.insert(1, "../cosyvoice/third_party/Matcha-TTS")
sys.path.insert(1, "../")

from cosyvoice.cli.cosyvoice import CosyVoice
from funasr import AutoModel
from config import MODELS_PATH
from llm import llm_model_chat
from process import clear_session
from custom_type import History

DS_API_TOKEN = os.getenv('DS_API_TOKEN')
dashscope.api_key = DS_API_TOKEN

tts_model_name_or_path = f"{MODELS_PATH}/cosyvoice/pretrained_models/CosyVoice-300M-Instruct"
asr_model_name_or_path = f"{MODELS_PATH}/sensevoice/SenseVoiceSmall"

speaker_name = '中文女'
cosyvoice_model = CosyVoice(tts_model_name_or_path)
# cosyvoice = CosyVoice2(f"{MODELS_PATH}/pretrained_models/CosyVoice2-0.5B")
# asr_model_name_or_path = "iic/SenseVoiceSmall"
sensevoice_model = AutoModel(model=asr_model_name_or_path,
                  vad_model="fsmn-vad",
                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True, 
                  device="cuda:0", 
                  remote_code="./sensevoice/model.py"
                  )

llm_model = "qwen2-72b-instruct"

def model_chat(audio, history: Optional[History]) -> Tuple[History, str, str]:
    generator_output = llm_model_chat(llm_model, sensevoice_model, cosyvoice_model, speaker_name, audio, history)
    
    (chatbot_output, audio_output, audio_input) = next(generator_output)
    
    return (chatbot_output, audio_output, audio_input)

os.makedirs("./tmp", exist_ok=True)

with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>AudioLLM-Voice Chat</center>""")

    chatbot = gr.Chatbot(label='AudioLLM')
    with gr.Row(): # 创建一个水平布局
        audio_input = gr.Audio(sources="microphone", label="Audio Input") # sources="microphone" 表示仅支持麦克风输入
        audio_output = gr.Audio(label="Audio Output", autoplay=True, streaming=True)
        clear_button = gr.Button("Clear")

    # 当用户停止录音时，触发 model_chat 函数。
    audio_input.stop_recording(model_chat, inputs=[audio_input, chatbot], outputs=[chatbot, audio_output, audio_input])
    clear_button.click(clear_session, outputs=[chatbot, audio_output, audio_input])


if __name__ == "__main__":
    demo.queue(api_open=False)
    demo.launch(server_name='0.0.0.0', 
                server_port=60001, 
                ssl_certfile="../cert.pem", 
                ssl_keyfile="../key.pem",
                inbrowser=True, 
                ssl_verify=False)