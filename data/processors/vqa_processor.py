import copy

from data.processors.box_processor import BOX_PROCESSORS


class VQAProcessor(object):
    """ VQA text processor, support format: 

        [{'from': 'human', 'value': '<image>\nWhat is the girl eating in the image?'}
         {'from': 'gpt', 'value': 'The girl in the image is eating a dessert, which appears to be a graham cracker treat or a cookie sandwich.'}
         {'from': 'human', 'value': "Describe the girl's hair color and clothing."}
         {'from': 'gpt', 'value': 'The girl has blonde hair, and she is wearing a pink shirt.'}]

         Or for dpo:
                [
                    {"from": "human", "value": "What is it?"},
                    {"from": "chosen", "value": "It is a bottle."},
                    {"from": "rejected", "value": "I don't know."},
                ]

    """

    SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. " + \
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "

    def __init__(self, key,
                 vision_placeholder='',
                 system_message=None,
                 roles=("USER: ", "ASSISTANT:"),
                 box_format="shikra",
                 ):
        self.key = key
        assert len(roles[0]) > 4, f"get roles = {roles}, check your setting."
        self.roles = roles
        self.vision_placeholder = vision_placeholder
        self.system_message = self.SYSTEM_MESSAGE if system_message is None else system_message
        self.box_processor = BOX_PROCESSORS[box_format]

    def preprocess(self, messages):
        # add media token in the first message if not exists
        if self.vision_placeholder not in messages[0]["value"]:
            messages[0]["value"] = self.vision_placeholder + messages[0]["value"]

        for _, message in enumerate(messages):

            if "box" in message:
                # reformat boxes
                message["value"] = self.box_processor(
                    message["value"], message["box"])

    def process_default(self, data_dict):
        messages = copy.deepcopy(data_dict.get(self.key, []))
        self.preprocess(messages)

        q_str_list, a_str_list = [], []

        for i in range(0, len(messages), 2):
            question = self.roles[0] + messages[i]["value"] + self.roles[1]

            if i == 0:
                question = self.system_message + question

            answer = messages[i + 1]["value"]
            q_str_list.append(question)
            a_str_list.append(answer)

        return q_str_list, a_str_list
    
    def __call__(self, data_dict):
        return self.process_default(data_dict)
