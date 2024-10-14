# whisper-infer
Script to run inference on whisper models using faster_whisper

**whisper-infer** employs [faster_whisper](https://github.com/SYSTRAN/faster-whisper/) to run the inference of Whisper models.

## Usage

### whisper-infer.py

```
$> python ./whisper-infer.py medium example.wav --split_stereo --output example.trs --load '{"device": "cuda", "compute_type": "int8"}' --transcribe '{"beam_size": 5, "word_timesta^Cs": false, "vad_filter": true}'
```

```python
from scripts.inference import infer

MODEL_NAME = 'medium'
AUDIO_FILE = 'example.wav'
load = {"device": "cuda", "compute_type": "int8"}
w = infer(MODEL_NAME, load=load)

transcribe = {"beam_size": 5, "word_timestaps": false, "vad_filter": true}
res = w(args.audio, fdo=fdo, split_stereo=args.split_stereo, transcribe=transcribe)
```

### eval-asr.py


### align-asr.py

