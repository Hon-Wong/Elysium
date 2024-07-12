import random

from data.processors.box_processor import BOX_PROCESSORS


class OnlineVQAProcessor(object):
    """online text processor, support construct prompt online"""

    SYSTEM_MESSAGE = (
        "A chat between a curious user and an artificial intelligence assistant. "
        + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )

    PREDEFINED_TASKS = ["VideoREC", "VideoREG", "SOT", "REC", "REG"]

    TEMPLATE_MAP = dict(
        VideoREC="./data/processors/templates/videorec_templates.txt",
        VideoREG="./data/processors/templates/videoreg_templates.txt",
        SOT="./data/processors/templates/sot_templates.txt",
        REC="./data/processors/templates/rec_templates.txt",
        REG="./data/processors/templates/reg_templates.txt",
    )

    BOX_STR_TEMPLATE = "Frame {i}: <box>"
    FRAME_STR_TEMPLATE = "Frame {i}: <image>"

    ANSWER_TEMPLATE = dict(
        VideoREC="{box_str}.",
        VideoREG="{object_description}.",
        SOT="{box_str}.",
        REC="{box_str}",
        REG="{expression}",
    )

    task2tag_map = {
        "REC": "[refer]",
        "REG": "[identify]",
    }

    def __init__(
        self,
        vision_placeholder="",
        system_message=None,
        box_format="ours_v1",
        box_key="box",
        fix_prompt=False,
        task="auto",
        enable_tag=False,
        roles={"human": "USER: ", "gpt": "ASSISTANT:"},
        single_frame_str="<image>\n",
    ) -> None:
        self.roles = roles
        self.media_token = vision_placeholder
        self.system_message = system_message or self.SYSTEM_MESSAGE
        self.box_processor = BOX_PROCESSORS[box_format]
        self.box_key = box_key
        self.fix_prompt = fix_prompt
        assert task in self.PREDEFINED_TASKS + ["auto"], f"Invalid task type {task}!"
        self.task_list = self.PREDEFINED_TASKS[:3] if task == "auto" else [task]
        self.question_candidates = {k: [] for k in self.task_list}
        self.enable_tag = enable_tag
        self.single_frame_str = single_frame_str

        for _task in self.task_list:
            with open(self.TEMPLATE_MAP[_task], "r") as f:
                for line in f.readlines():
                    self.question_candidates[_task].append(line.strip())
                    if self.fix_prompt:
                        break

    def construct_messages(self, data_dict, task):
        frame_len = len(data_dict[self.box_key])
        if frame_len > 1:
            frame_str = (
                ", ".join(
                    self.FRAME_STR_TEMPLATE.format(**{"i": i + 1})
                    for i in range(0, frame_len)
                )
                + "\n"
            )
            box_str = ", ".join(
                self.BOX_STR_TEMPLATE.format(**{"i": i + 1})
                for i in range(0, frame_len)
            )

        else:
            frame_str = self.single_frame_str
            box_str = "<box>"

        tag = self.task2tag_map[task] + " " if self.enable_tag else ""
        data_dict["box_str"] = box_str
        question_str = (
            self.system_message
            + self.roles["human"]
            + frame_str
            + tag
            + random.choice(self.question_candidates[task]).format(**data_dict)
            + self.roles["gpt"]
        )
        answer_str = self.ANSWER_TEMPLATE[task].format(**data_dict)

        if self.box_processor.box_token in question_str:
            if question_str.count(self.box_processor.box_token) == 1:
                question_str = self.box_processor(
                    question_str, [data_dict[self.box_key][0]]
                )
            else:
                question_str = self.box_processor(question_str, data_dict[self.box_key])

        if self.box_processor.box_token in answer_str:
            if answer_str.count(self.box_processor.box_token) == 1:
                answer_str = self.box_processor(
                    answer_str, [data_dict[self.box_key][0]]
                )
            else:
                answer_str = self.box_processor(answer_str, data_dict[self.box_key])

        return question_str, answer_str

    def __call__(self, data_dict):
        task = random.choice(self.task_list)
        data_dict["vid"] = f"{task}_" + str(data_dict["vid"])
        question, answer = self.construct_messages(data_dict, task)
        return [question], [answer]
