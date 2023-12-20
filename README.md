# Auto

Live lane detection, stabilized steering, and object recognition all in one. ([YouTube Playlist](https://www.youtube.com/watch?v=I9cR4of2jlo&list=PL4iMkUwfSFa3LzvXLPlDLKk0EshU3x4gV))

https://github.com/KoralK5/Auto/assets/62809012/2b1f068a-b666-45b0-9d4e-7942cb7735b6

## Table of Contents

- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)

## Overview

To run the live lane detection on a video,

1. Place the video you want to use in the Videos directory.
2. Place the `.weights` of the model you want to use in the Data directory. This can be downloaded [here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights). Note that YOLOv4-tiny was used for the demo video.
3. `pip install requirements.txt`.
4. Run `autonomous.py` or `oldCode.py`.

## Requirements

List of dependencies and requirements needed to run the project.

- Python (version 3)
- OpenCV (version 4)
- Numpy (version 1)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/KoralK5/Auto.git
```
