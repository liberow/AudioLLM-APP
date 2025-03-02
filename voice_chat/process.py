def preprocess(text: str):
    """对文本进行预处理，按标点符号分割句子，并合并过短的句子。"""
    seperators = ['.', '。', '?', '!']
    min_sentence_len = 10
    # split sentence
    seperator_index = [i for i, j in enumerate(text) if j in seperators] # 分隔符在 text 中的 index
    if len(seperator_index) == 0:
        return [text]
    # texts = [text[:seperator_index[i] + 1] if i == 0 else text[seperator_index[i - 1] + 1: seperator_index[i] + 1] for i in range(len(seperator_index))]
    texts = []
    for i in range(len(seperator_index)):
        if i == 0:
            texts.append(text[:seperator_index[i] + 1])
        else:
            texts.append(text[seperator_index[i - 1] + 1: seperator_index[i] + 1])    
    remains = text[seperator_index[-1] + 1:]
    if len(remains) != 0:
        texts.append(remains)
    # merge short sentence
    texts_merge = []
    this_text = texts[0]
    for i in range(1, len(texts)):
        if len(this_text) >= min_sentence_len:
            texts_merge.append(this_text)
            this_text = texts[i]
        else:
            this_text += texts[i]
    texts_merge.append(this_text)
    return texts_merge


def clear_session():
    return ([], None, None)


