class BoxProcessor(object):
    def __init__(
        self, box_token="<box>", value_format=".3f", value_sep=", ", bounds=("[", "]")
    ):
        self.box_token = box_token
        self.sep = value_sep
        self.value_format = value_format
        self.bounds = bounds

    def _format_value(self, value):
        if self.value_format == ".3f":
            return f"{value:.3f}"
        elif self.value_format == ".2f":
            return f"{value:.2f}"
        elif self.value_format == "2d":
            value = int(round(value * 100))
            return str(value)
        elif self.value_format == "3d":
            value = int(round(value * 1000))
            return str(value)
        elif self.value_format == "<2d>":
            value = int(round(value * 100))
            return f"<{value:d}>"
        elif self.value_format == "<3d>":
            value = int(round(value * 1000))
            return f"<{value:d}>"
        else:
            raise NotImplementedError(f'Invalid value format: "{self.value_format}"')

    def _format_box(self, box):
        return (
            self.bounds[0]
            + self.sep.join([self._format_value(pos) for pos in box])
            + self.bounds[1]
        )

    def __call__(self, input_str, boxes):
        box_strings = [self._format_box(box) for box in boxes]

        num_boxes = input_str.count(self.box_token)
        if num_boxes == 0:
            return input_str

        assert (
            num_boxes == len(box_strings)
        ), f"Error: Number of {self.box_token} tags does not match number of list elements."

        # replace <box> token with formatted box
        for box_str in box_strings:
            input_str = input_str.replace(self.box_token, box_str, 1)
        return input_str


BOX_PROCESSORS = {
    "shikra": BoxProcessor(value_format=".3f", value_sep=","),
    "cog_vlm": BoxProcessor(value_format="3d", value_sep=",", bounds=("[[", "]]")),
    "minigpt_v2": BoxProcessor(value_format="<2d>", value_sep="", bounds=("{", "}")),
    "ours_v1": BoxProcessor(value_format="2d", value_sep=","),
}


if __name__ == "__main__":
    input_str = (
        "Please tell me more about the rectangular section <box> in the photo <image>."
    )
    box = [[0.042, 0.196, 0.152, 0.454]]
    for name, processor in BOX_PROCESSORS.items():
        print("processor:", name)
        print("result:", processor(input_str, box))

    input_str = "Please tell me more about the rectangular section <point> in the photo <image>."
    point = [[0.042, 0.196]]
    processor = BoxProcessor(value_format="3d", value_sep=", ", box_token="<point>")
    print("point:")
    print(processor(input_str, boxes=point))
