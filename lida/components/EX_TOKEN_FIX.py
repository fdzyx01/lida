from dataclasses import asdict

import tiktoken

# jetbrains://pycharm/navigate/reference?project=lida&path=C:/PythonVolVenv/metaagent10/Lib/site-packages/llmx/utils.py
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if (
            model == "gpt-3.5-turbo-0301" or True
    ):  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                message = asdict(message)

            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )

            for key, value in message.items():
                if type(value) == str:
                    target = value
                elif type(value) == dict and value["type"] == "text":
                    target = value["content"]
                else:
                    target = ""
                    print("警告：字段非文字，无法计算token。" + str(value))
                if target:
                    num_tokens += len(encoding.encode(target))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
