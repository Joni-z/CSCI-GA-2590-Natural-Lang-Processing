import re


def your_netid():
    YOUR_NET_ID = "zz5070"
    return YOUR_NET_ID


def your_hf_token():
    return "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"


def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers."""
    prefix = (
        "Add the two 7-digit integers exactly. "
        "Output only the final sum digits on the first line.\n"
        "Question: What is 1234567+1234567?\n"
        "Answer: 2469134\n"
        "Question: What is 7654321+1111111?\n"
        "Answer: 8765432\n"
        "Question: What is 9999999+1010101?\n"
        "Answer: 11010100\n"
        "Question: What is 1010101+9999999?\n"
        "Answer: 11010100\n"
        "Question: What is 8888888+2020202?\n"
        "Answer: 10909090\n"
        "Question: What is 9596979+9293949?\n"
        "Answer: 18890928\n"
        "Question: What is 9847464+5252529?\n"
        "Answer: 15099993\n"
        "Question: What is 3156283+4084597?\n"
        "Answer: 7240880\n"
        "Question: What is 9000100+1000010?\n"
        "Answer: 10000110\n"
        "Question: What is 6732655+3428081?\n"
        "Answer: 10160736\n"
        "Question: What is 1368762+1148749?\n"
        "Answer: 2517511\n"
        "Question: What is 8319432+9214800?\n"
        "Answer: 17534232\n"
        "Question: What is 6459722+7565469?\n"
        "Answer: 14025191\n"
        "Question: What is 5892625+9415651?\n"
        "Answer: 15308276\n"
        "Question: What is 8342682+1024647?\n"
        "Answer: 9367329\n"
        "Question: What is 8716638+1302271?\n"
        "Answer: 10018909\n"
        "Question: What is 7566899+2580901?\n"
        "Answer: 10147800\n"
        "Question: What is 8768610+5873151?\n"
        "Answer: 14641761\n"
        "Question: What is 3697407+4804185?\n"
        "Answer: 8501592\n"
        "Question: What is 1254878+2118550?\n"
        "Answer: 3373428\n"
        "Question: What is 7035620+6824705?\n"
        "Answer: 13860325\n"
        "Question: What is 6538466+4453098?\n"
        "Answer: 10991564\n"
        "Question: What is 9974889+9827518?\n"
        "Answer: 19802407\n"
        "Question: What is 7169878+6854133?\n"
        "Answer: 14024011\n"
        "Question: What is 7196020+4500293?\n"
        "Answer: 11696313\n"
        "Question: What is 2215869+7493395?\n"
        "Answer: 9709264\n"
        "Question: What is 5728189+3792177?\n"
        "Answer: 9520366\n"
        "Question: What is 5372518+9005390?\n"
        "Answer: 14377908\n"
        "Question: What is 9406391+4220157?\n"
        "Answer: 13626548\n"
        "Question: What is 6143768+3896825?\n"
        "Answer: 10040593\n"
        "Question: What is 6348700+4041201?\n"
        "Answer: 10389901\n"
        "Question: What is 4524571+9012469?\n"
        "Answer: 13537040\n"
        "Question: What is 3044419+6608684?\n"
        "Answer: 9653103\n"
        "Question: What is 1756139+8493797?\n"
        "Answer: 10249936\n"
        "Question: What is "
    )
    suffix = "?\nAnswer: "
    return prefix, suffix


def your_config():
    """Returns a config for prompting api."""
    config = {
        "max_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.6,
        "repetition_penalty": 1,
        "stop": [],
    }
    return config


def your_pre_processing(s):
    return s.strip()


def your_post_processing(output_string):
    """Extract the answer from the first generated line without doing arithmetic."""
    cleaned = output_string.replace(",", "").strip()
    if not cleaned:
        return 0

    first_line = cleaned.splitlines()[0].strip()
    tagged = re.search(r"Answer:\s*(\d+)", first_line)
    if tagged:
        token = tagged.group(1)
        if 7 <= len(token) <= 8:
            return int(token)
        if len(token) > 8:
            if token.startswith("1"):
                return int(token[:8])
            return int(token[:7])

    number_spans = re.findall(r"\d+", first_line)
    for token in number_spans:
        if 7 <= len(token) <= 8:
            return int(token)
        if len(token) > 8:
            if token.startswith("1"):
                return int(token[:8])
            return int(token[:7])

    tagged_anywhere = re.search(r"Answer:\s*(\d+)", cleaned)
    if tagged_anywhere:
        token = tagged_anywhere.group(1)
        if len(token) > 8:
            if token.startswith("1"):
                return int(token[:8])
            return int(token[:7])
        return int(token)

    any_number = re.search(r"\d+", cleaned)
    if any_number:
        token = any_number.group(0)
        if len(token) > 8:
            if token.startswith("1"):
                return int(token[:8])
            return int(token[:7])
        return int(token)

    return 0
