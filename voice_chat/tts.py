import re

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from process import preprocess

def text_to_speech_zero_shot(
        tts_model: CosyVoice, 
        text: str, 
        prompt_text: str, 
        audio_prompt_path: str, 
        speaker_name: str):
    prompt_speech_16k = load_wav(audio_prompt_path, 16000)
    pattern = r"生成风格:\s*([^;]+);播报内容:\s*(.+)"
    match = re.search(pattern, text)
    if match:
        style = match.group(1).strip()
        content = match.group(2).strip()
        tts_text = f"{content}"
        prompt_text = f"{style}<endofprompt>{prompt_text}"
        print(f"生成风格: {style}")
        print(f"播报内容: {content}")
    else:
        print("No match found")
        tts_text = text

    text_list = preprocess(tts_text)
    for i in text_list:
        output = tts_model.inference_sft(i, speaker_name)
        for chunk in output:  
            if 'tts_speech' in chunk:  # 确保输出中包含 'tts_speech'
                yield (22050, chunk['tts_speech'].numpy().flatten())
            else:
                print("Warning: 'tts_speech' not found in output")   


def text_to_speech(tts_model: CosyVoice, text: str, speaker_name: str):
    pattern = r"生成风格:\s*([^;]+);播报内容:\s*(.+)"
    match = re.search(pattern, text)
    if match:
        style = match.group(1).strip()
        content = match.group(2).strip()
        tts_text = f"{style}<endofprompt>{content}"
        print(f"\n生成风格: {style}\n")
        print(f"\n播报内容: {content}\n")
    else:
        print("No match found")
        tts_text = text

    text_list = preprocess(tts_text)
    for i in text_list:
        output = tts_model.inference_sft(i, speaker_name)
        for chunk in output:  
            if 'tts_speech' in chunk:  # 确保输出中包含 'tts_speech'
                yield (22050, chunk['tts_speech'].numpy().flatten())
            else:
                print("Warning: 'tts_speech' not found in output")    
