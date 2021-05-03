def restore_qnli(input_text):
    question, sentence = input_text.split(" [SEP] ")
    question = question[10:] # remove "question: "
    sentence = sentence[10:] # remove "sentence: "
    return question, sentence

def qnli_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    question, sentence = restore_qnli(in_text)

    in_text_new = "{} <mask>. Yes, {}".format(question, sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "Okay")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Nonetheless")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template1_backward(in_text, out_text):
    if "Okay. Yes," in out_text:
        out_text_old = "entailment"
    elif "Nonetheless. Yes," in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def qnli_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    question, sentence = restore_qnli(in_text)

    in_text_new = "{} <mask>. It is known that, {}".format(question, sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "Notably")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Yet")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template2_backward(in_text, out_text):
    if "Notably. It is known that," in out_text:
        out_text_old = "entailment"
    elif "Yet. It is known that," in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def qnli_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    question, sentence = restore_qnli(in_text)

    in_text_new = "{} <mask>, however, {}".format(question, sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "Specifically")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Notably")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template3_backward(in_text, out_text):
    if "Specifically, however," in out_text:
        out_text_old = "entailment"
    elif "Notably, however," in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

TEMPLATES = {
    "qnli_t1": (qnli_template1_forward, qnli_template1_backward),
    "qnli_t2": (qnli_template2_forward, qnli_template2_backward),
    "qnli_t3": (qnli_template3_forward, qnli_template3_backward)
}