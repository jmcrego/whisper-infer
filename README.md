# whisper-infer
Python wrapper for faster_whisper to run inference on Whiseper models.

**whisper-infer** employs [faster_whisper](https://github.com/SYSTRAN/faster-whisper/) to run the inference of Whisper models.

## Usage

### whisper-infer.py

Use the next code to run ASR inference over an audio file:
```python
from scripts.inference import infer

MODEL_NAME = 'medium'
AUDIO_FILE = 'example.wav'
OUTPUT_FILE = 'example.trs'
SPLIT_STEREO = True
LOAD = {"device": "cuda", "compute_type": "int8"}
TRANSCRIBE = {"beam_size": 5, "word_timestaps": false, "vad_filter": true}

w = infer(MODEL_NAME, load=LOAD)

with open(OUTPUT_FILE, 'w') as fdo:
     for l in w(AUDIO_FILE, split_stereo=SPLIT_STEREO, transcribe=TRANSCRIBE):
     	 fdo.write(l + '\n')
```

Or use:
```
$> python ./whisper-infer.py medium example.wav --split_stereo --output example.trs --load '{"device": "cuda", "compute_type": "int8"}' --transcribe '{"beam_size": 5, "word_timesta^Cs": false, "vad_filter": true}'
```

### eval-asr.py


### align-asr.py

