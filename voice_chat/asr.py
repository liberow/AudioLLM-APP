import torch
import torchaudio
from uuid import uuid4
from funasr import AutoModel

def transcribe(asr_model: AutoModel, audio):
    sample_rate, data = audio
    file_path = f"./tmp/asr_{uuid4()}.wav"

    torchaudio.save(file_path, torch.from_numpy(data).unsqueeze(0), sample_rate)

    res = asr_model.generate(
        input=file_path,
        cache={},
        language="zh",
        text_norm="woitn",
        batch_size_s=0,
        batch_size=1
    )
    text = res[0]['text']
    res_dict = {"file_path": file_path, "text": text}
    print(res_dict)
    return res_dict
