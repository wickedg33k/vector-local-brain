# Third-party notices

## wire-pod

`wire-pod-patches/patches/*.patch` modify and add files in
[wire-pod](https://github.com/kercre123/wire-pod). The patch series is
distributed under the same MIT license. Upstream license (wire-pod `LICENSE`):

```
MIT License

Copyright (c) 2022 Kerigan Creighton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## chipper (Digital Dream Labs)

wire-pod's `chipper/` directory, which the patches touch, is derived from
Digital Dream Labs' open-sourced chipper. Upstream license (`chipper/LICENSE`):

```
MIT License

Copyright (c) 2021 Digital Dream Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not vendored (downloaded or installed at setup time)

| Component | Used by | License |
|---|---|---|
| [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) `face_detection_yunet_2023mar.onnx` | `faces/` | MIT (© 2020 Shiqi Yu) |
| [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) `face_recognition_sface_2021dec.onnx` | `faces/` | Apache-2.0 |
| [OpenCV](https://github.com/opencv/opencv-python) | `faces/` | Apache-2.0 (opencv-python-headless wheel: MIT packaging, bundles Apache-2.0 OpenCV) |
| [wirepod_vector_sdk](https://github.com/kercre123/wirepod-vector-python-sdk) / [anki_vector](https://github.com/digital-dream-labs/vector-python-sdk) | `curious/`, `faces/`, `music/` | Apache-2.0 |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | `music/` | Unlicense |
| [ytmusicapi](https://github.com/sigma67/ytmusicapi) | `music/` (optional) | MIT |
| [pycoral / libedgetpu / tflite-runtime](https://coral.ai/) + `google-coral/test_data` models | `coral/` | Apache-2.0 |
| [Vosk](https://alphacephei.com/vosk/) models | wire-pod STT | Apache-2.0 |
| [Ollama](https://github.com/ollama/ollama) and the models it serves | everything | MIT (Ollama); each model has its own license (Qwen3: Apache-2.0, Gemma 3: Gemma Terms of Use) |
