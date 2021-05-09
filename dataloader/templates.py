# turns out writing individual functions is not a good idea...

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
            out_text_new = "Okay"
        elif out_text == "not_entailment":
            out_text_new = "Nonetheless"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "okay":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "nonetheless":
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
            out_text_new = "Notably"
        elif out_text == "not_entailment":
            out_text_new = "Yet"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "notably":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "yet":
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
            out_text_new = "Specifically"
        elif out_text == "not_entailment":
            out_text_new = "Notably"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qnli_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "specifically":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "notably":
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
            out_text_new = "one"
        elif out_text == "unacceptable":
            out_text_new = "proof"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "one":
        out_text_old = "acceptable"
    elif out_text.strip().lower() == "proof":
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
            out_text_new = "wrong"
        elif out_text == "unacceptable":
            out_text_new = "sad"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "wrong":
        out_text_old = "acceptable"
    elif out_text.strip().lower() == "sad":
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
            out_text_new = "misleading"
        elif out_text == "unacceptable":
            out_text_new = "disappointing"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def cola_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "misleading":
        out_text_old = "acceptable"
    elif out_text.strip().lower() == "disappointing":
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
            out_text_new = "Clearly"
        elif out_text == "not_entailment":
            out_text_new =  "Yet"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "clearly":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "yet":
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
            out_text_new = "Accordingly"
        elif out_text == "not_entailment":
            out_text_new = "Meanwhile"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "accordingly":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "meanwhile":
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
            out_text_new = "So"
        elif out_text == "not_entailment":
            out_text_new = "Meanwhile"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def rte_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "so":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "meanwhile":
        out_text_old = "not_entailment"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def restore_sst2(input_text):
    return input_text[10:]

def sst2_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    in_text = restore_sst2(in_text)

    in_text_new = "{} A <mask> one.".format(in_text)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "positive":
            out_text_new = "irresistible"
        elif out_text == "negative":
            out_text_new =  "pathetic"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def sst2_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "irresistible":
        out_text_old = "positive"
    elif out_text.strip().lower() == "pathetic":
        out_text_old = "negative"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def sst2_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    in_text = restore_sst2(in_text)

    in_text_new = "{} A <mask> piece.".format(in_text)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "positive":
            out_text_new = "wonderful"
        elif out_text == "negative":
            out_text_new =  "bad"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def sst2_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "wonderful":
        out_text_old = "positive"
    elif out_text.strip().lower() == "bad":
        out_text_old = "negative"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def sst2_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    in_text = restore_sst2(in_text)

    in_text_new = "{} All in all <mask>.".format(in_text)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "positive":
            out_text_new = "delicious"
        elif out_text == "negative":
            out_text_new =  "bad"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def sst2_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "delicious":
        out_text_old = "positive"
    elif out_text.strip().lower() == "bad":
        out_text_old = "negative"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mnli_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte
    hypo = hypo[0].lower() + hypo[1:]

    in_text_new = "{} <mask>, you are right, {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "Fine"
        elif out_text == "neutral":
            out_text_new =  "Plus"
        elif out_text == "contradiction":
            out_text_new = "Otherwise"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mnli_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "fine":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "plus":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "otherwise":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mnli_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte
    hypo = hypo[0].lower() + hypo[1:]

    in_text_new = "{} <mask> you're right {}.".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "There"
        elif out_text == "neutral":
            out_text_new =  "Plus"
        elif out_text == "contradiction":
            out_text_new = "Otherwise"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mnli_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "there":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "plus":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "otherwise":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mnli_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte

    in_text_new = "{} <mask>! {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "Meaning"
        elif out_text == "neutral":
            out_text_new =  "Plus"
        elif out_text == "contradiction":
            out_text_new = "Otherwise"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mnli_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "meaning":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "plus":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "otherwise":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def snli_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte
    hypo = hypo[0].lower() + hypo[1:]

    in_text_new = "{} <mask>, no, {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "Alright"
        elif out_text == "neutral":
            out_text_new =  "Watch"
        elif out_text == "contradiction":
            out_text_new = "Except"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def snli_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "alright":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "watch":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "except":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def snli_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte

    in_text_new = "{} <mask>, in this case {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "Hi"
        elif out_text == "neutral":
            out_text_new =  "Watch"
        elif out_text == "contradiction":
            out_text_new = "Worse"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def snli_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "hi":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "watch":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "worse":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def snli_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte
    hypo = hypo[0].lower() + hypo[1:]

    in_text_new = "{} <mask> this time {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "entailment":
            out_text_new = "Regardless"
        elif out_text == "neutral":
            out_text_new =  "Fortunately"
        elif out_text == "contradiction":
            out_text_new = "Unless"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def snli_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "regardless":
        out_text_old = "entailment"
    elif out_text.strip().lower() == "fortunately":
        out_text_old = "neutral"
    elif out_text.strip().lower() == "unless":
        out_text_old = "contradiction"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mrpc_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte

    in_text_new = "{} <mask>! {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "equivalent":
            out_text_new = "Rather"
        elif out_text == "not_equivalent":
            out_text_new =  "Alas"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mrpc_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "rather":
        out_text_old = "equivalent"
    elif out_text.strip().lower() == "alas":
        out_text_old = "not_equivalent"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mrpc_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte
    hypo = hypo[0].lower() + hypo[1:]

    in_text_new = "{} <mask>. This is the first time {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "equivalent":
            out_text_new = "At"
        elif out_text == "not_equivalent":
            out_text_new =  "Thus"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mrpc_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "at":
        out_text_old = "equivalent"
    elif out_text.strip().lower() == "thus":
        out_text_old = "not_equivalent"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def mrpc_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    premise, hypo = restore_rte(in_text) # reuse rte

    in_text_new = "{} <mask>. That's right. {}".format(premise, hypo)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "equivalent":
            out_text_new = "Instead"
        elif out_text == "not_equivalent":
            out_text_new =  "Moreover"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def mrpc_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "instead":
        out_text_old = "equivalent"
    elif out_text.strip().lower() == "moreover":
        out_text_old = "not_equivalent"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def restore_qqp(input_text):
    q1, q2 = input_text.split(" [SEP] ")
    q1 = q1[11:].strip() # remove "question1: "
    q2 = q2[11:].strip() # remove "sentence1: "
    return q1, q2

def qqp_template1_forward(in_text, out_texts):
    # out_texts is a list of out_text
    q1, q2 = restore_qqp(in_text) # reuse rte
    q2 = q2[0].lower() + q2[1:]

    in_text_new = "{} <mask>, but {}".format(q1, q2)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "duplicate":
            out_text_new = "Me"
        elif out_text == "not_duplicate":
            out_text_new =  "Since"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qqp_template1_backward(in_text, out_text):
    if out_text.strip().lower() == "me":
        out_text_old = "duplicate"
    elif out_text.strip().lower() == "since":
        out_text_old = "not_duplicate"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def qqp_template2_forward(in_text, out_texts):
    # out_texts is a list of out_text
    q1, q2 = restore_qqp(in_text) # reuse rte
    q2 = q2[0].lower() + q2[1:]

    in_text_new = "{} <mask>, please, {}".format(q1, q2)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "duplicate":
            out_text_new = "Um"
        elif out_text == "not_duplicate":
            out_text_new =  "Best"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qqp_template2_backward(in_text, out_text):
    if out_text.strip().lower() == "um":
        out_text_old = "duplicate"
    elif out_text.strip().lower() == "best":
        out_text_old = "not_duplicate"
    else:
        out_text_old = "unknown"
    return "", out_text_old

def qqp_template3_forward(in_text, out_texts):
    # out_texts is a list of out_text
    q1, q2 = restore_qqp(in_text) # reuse rte
    q2 = q2[0].lower() + q2[1:]

    in_text_new = "{} <mask>, I want to know {}".format(q1, q2)
    out_texts_new = []

    for out_text in out_texts:
        if out_text == "duplicate":
            out_text_new = "Ironically"
        elif out_text == "not_duplicate":
            out_text_new =  "Beyond"
        else:
            raise Exception
        out_texts_new.append(out_text_new)

    return in_text_new, out_texts_new

def qqp_template3_backward(in_text, out_text):
    if out_text.strip().lower() == "ironically":
        out_text_old = "duplicate"
    elif out_text.strip().lower() == "beyond":
        out_text_old = "not_duplicate"
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
    "sst2_t1": (sst2_template1_forward, sst2_template1_backward),
    "sst2_t2": (sst2_template2_forward, sst2_template2_backward),
    "sst2_t3": (sst2_template3_forward, sst2_template3_backward),
    "mnli_t1": (mnli_template1_forward, mnli_template1_backward),
    "mnli_t2": (mnli_template2_forward, mnli_template2_backward),
    "mnli_t3": (mnli_template3_forward, mnli_template3_backward),
    "snli_t1": (snli_template1_forward, snli_template1_backward),
    "snli_t2": (snli_template2_forward, snli_template2_backward),
    "snli_t3": (snli_template3_forward, snli_template3_backward),
    "mrpc_t1": (mrpc_template1_forward, mrpc_template1_backward),
    "mrpc_t2": (mrpc_template2_forward, mrpc_template2_backward),
    "mrpc_t3": (mrpc_template3_forward, mrpc_template3_backward),
    "qqp_t1": (qqp_template1_forward, qqp_template1_backward),
    "qqp_t2": (qqp_template2_forward, qqp_template2_backward),
    "qqp_t3": (qqp_template3_forward, qqp_template3_backward),
}