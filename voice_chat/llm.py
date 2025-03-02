import re
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import Optional, Generator
from http import HTTPStatus

from custom_type import History
from asr import transcribe
from prompt_template import default_system
from messages import history_to_messages, messages_to_history
from tts import text_to_speech

def llm_model_chat(llm_model: str, asr_model: str, tts_model: str, speaker_name: str, audio, history: Optional[History]
               ) -> Generator[History, str, str]:
    if audio is None:
        query = ''
        asr_wav_path = None
    else:
        asr_res = transcribe(asr_model, audio)
        query, asr_wav_path = asr_res['text'], asr_res['file_path']
    if history is None:
        history = []
    system = default_system
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    print(messages)
    gen = Generation()
    llm_stream = False
    if llm_stream:
        gen = gen.call(
            llm_model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            enable_search=False,
            stream=llm_stream,
        )
    else:
        gen = [gen.call(
            llm_model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            enable_search=False,
            stream=llm_stream
        )]
    processed_tts_text = ""
    punctuation_pattern = r'([!?;。！？])'
    for response in gen:
        if response.status_code == HTTPStatus.OK:
            role = response.output.choices[0].message.role
            response = response.output.choices[0].message.content
            print(f"response: {response}")
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            # 对 processed_tts_text 进行转义处理
            escaped_processed_tts_text = re.escape(processed_tts_text)
            tts_text = re.sub(f"^{escaped_processed_tts_text}", "", response)
            if re.search(punctuation_pattern, tts_text):
                parts = re.split(punctuation_pattern, tts_text)
                if len(parts) > 2 and parts[-1] and llm_stream: # parts[-1]为空说明句子以标点符号结束，没必要截断
                    tts_text = "".join(parts[:-1])
                print(f"processed_tts_text: {processed_tts_text}")
                processed_tts_text += tts_text
                print(f"cur_tts_text: {tts_text}")
                tts_generator = text_to_speech(tts_model, tts_text, speaker_name)
                # tts_generator = text_to_speech_zero_shot(tts_text, query, asr_wav_path)
                for output_audio_path in tts_generator:
                    yield (history, output_audio_path, None)
        else:
            raise ValueError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    if processed_tts_text == response:
        print("turn end")
    else:
        escaped_processed_tts_text = re.escape(processed_tts_text)
        tts_text = re.sub(f"^{escaped_processed_tts_text}", "", response)
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech(tts_text)
        # tts_generator = text_to_speech_zero_shot(tts_text, query, asr_wav_path)
        for output_audio_path in tts_generator:
            yield (history, output_audio_path, None)
        processed_tts_text += tts_text
        print(f"processed_tts_text: {processed_tts_text}")
        print("turn end")
