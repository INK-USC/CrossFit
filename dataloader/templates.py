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

def cola_template1_forward(in_text, out_texts):
    sentence = in_text

    in_text_new = "{} You are <mask>.".format(sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "acceptable":
            out_text_new = in_text_new.replace("<mask>", "one")
        elif out_text == "unacceptable":
            out_text_new = in_text_new.replace("<mask>", "proof")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template1_backward(in_text, out_text):
    if "You are one." in out_text:
        out_text_old = "acceptable"
    elif "You are proof." in out_text:
        out_text_old = "unacceptable"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def cola_template2_forward(in_text, out_texts):
    sentence = in_text

    in_text_new = "It is <mask>. {}".format(sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "acceptable":
            out_text_new = in_text_new.replace("<mask>", "wrong")
        elif out_text == "unacceptable":
            out_text_new = in_text_new.replace("<mask>", "sad")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template2_backward(in_text, out_text):
    if "It is wrong." in out_text:
        out_text_old = "acceptable"
    elif "It is sad." in out_text:
        out_text_old = "unacceptable"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def cola_template3_forward(in_text, out_texts):
    sentence = in_text

    in_text_new = "I am <mask>. {}".format(sentence)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "acceptable":
            out_text_new = in_text_new.replace("<mask>", "misleading")
        elif out_text == "unacceptable":
            out_text_new = in_text_new.replace("<mask>", "disappointing")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template3_backward(in_text, out_text):
    if "I am misleading." in out_text:
        out_text_old = "acceptable"
    elif "I am disappointing." in out_text:
        out_text_old = "unacceptable"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def restore_rte(input_text):
    premise, hypo = input_text.split(" [SEP] ")
    if premise.startswith("premise: "):
        premise = premise[9:]
    if premise.startswith("sentence 1: "):
        premise = premise[12:]
    if hypo.startswith("hypothesis: "):
        hypo = hypo[12:]
    if hypo.startswith("sentence 2: "):
        hypo = hypo[12:]
    hypo = hypo[:1].lower() + hypo[1:] # lower the first letter
    return premise, hypo

def rte_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text)

    in_text_new = "{} <mask>, I believe {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "Clearly")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Yet")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template1_backward(in_text, out_text):
    if "Clearly, I believe" in out_text:
        out_text_old = "entailment"
    elif "Yet, I believe" in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def rte_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text)

    in_text_new = "{} <mask>, I think that {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "Accordingly")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Meanwhile")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template2_backward(in_text, out_text):
    if "Accordingly, I think that" in out_text:
        out_text_old = "entailment"
    elif "Meanwhile, I think that" in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old


def rte_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text)

    in_text_new = "{} <mask>, I think {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = in_text_new.replace("<mask>", "So")
        elif out_text == "not_entailment":
            out_text_new = in_text_new.replace("<mask>", "Meanwhile")
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template3_backward(in_text, out_text):
    if "So, I think" in out_text:
        out_text_old = "entailment"
    elif "Meanwhile, I think" in out_text:
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

TEMPLATES = {
    "qnli_t1": (qnli_template1_forward, qnli_template1_backward),
    "qnli_t2": (qnli_template2_forward, qnli_template2_backward),
    "qnli_t3": (qnli_template3_forward, qnli_template3_backward),
    "cola_t1": (cola_template1_forward, cola_template1_backward),
    "cola_t2": (cola_template2_forward, cola_template2_backward),
    "cola_t3": (cola_template3_forward, cola_template3_backward),
    "rte_t1": (rte_template1_forward, rte_template1_backward),
    "rte_t2": (rte_template2_forward, rte_template2_backward),
    "rte_t3": (rte_template3_forward, rte_template3_backward),
}