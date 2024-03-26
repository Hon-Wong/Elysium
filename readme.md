# Elysium: Exploring Object-level Perception in Videos via MLLM

**MLLM can recognize and track anything in videos now!**

<a href='https://arxiv.org/abs/2403.16558'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elysium-exploring-object-level-perception-in/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=elysium-exploring-object-level-perception-in)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elysium-exploring-object-level-perception-in/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=elysium-exploring-object-level-perception-in)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elysium-exploring-object-level-perception-in/zeroshot-video-question-answer-on-tgif-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-tgif-qa?p=elysium-exploring-object-level-perception-in)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elysium-exploring-object-level-perception-in/zero-shot-single-object-tracking-on-lasot)](https://paperswithcode.com/sota/zero-shot-single-object-tracking-on-lasot?p=elysium-exploring-object-level-perception-in)


## Abstract

Multi-modal Large Language Models (MLLMs) have demonstrated their ability to perceive objects in still images, but their application in video-related tasks, such as object tracking, remains understudied. This lack of exploration is primarily due to two key challenges. Firstly, extensive pretraining on large-scale video datasets is required to equip MLLMs with the capability to perceive objects across multiple frames and understand inter-frame relationships. Secondly, processing a large number of frames within the context window of Large Language Models (LLMs) can impose a significant computational burden.
To address the first challenge, we introduce ElysiumTrack-1M, a large-scale video dataset paired with novel tasks: Referring Single Object Tracking (RSOT) and Video Referring Expression Generation (Video-REG). ElysiumTrack-1M contains 1.27 million annotated video frames with corresponding object boxes and descriptions. Leveraging this dataset, we conduct training of MLLMs and propose a token-compression model T-Selector to tackle the second challenge. Our proposed approach, Elysium: Exploring Object-level Perception in Videos via MLLM, is an end-to-end trainable MLLM that makes the first attempt to conduct object-level tasks in videos without requiring any additional plug-in or expert models.

## Demo Videos

**Referring Single Object Tracking (RSOT)**

We use prompt "Please find {expression} in the initial frame and provide the detailed coordinates in each frame." for each video.

| ![GIF 1](demo/a_running_dog_played_in_the_snow_field.gif) | ![GIF 2](demo/the_cap_on_a_dogs_head.gif) | ![GIF 3](demo/the_snow_field.gif) | ![GIF 4](demo/shoes.gif) | ![GIF 5](demo/the_person_in_red.gif) |
|---|---|---|---|---|
| a running dog played in the snow field | the cap on a dog's head | the snow field | shoes | the person in red |

| ![GIF 6](demo/boy_back_to_camera.gif) | ![GIF 7](demo/a_dancing_kangaroo.gif) | ![GIF 8](demo/dog.gif) |
|---|---|---|
| boy back to camera | a dancing kangaroo | dog |

**Single Object Tracking (SOT)**

We use prompt "This is a video showing an object with coordinates {coordinates} in Frame 1. Provide the detailed coordinates of the object in each frame." for each video.

| ![Dog Coordinates](demo/coords_dog.gif) | ![Airplane Coordinates](demo/coords_airplane.gif) |
|---|---|
| [34,40,51,67] | [35,48,60,55] |
