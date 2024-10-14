# whisper-infer
Python wrapper for [faster_whisper](https://github.com/SYSTRAN/faster-whisper/) to run inference on Whisper models.

## Usage

### whisper-infer.py

Use the next code to run ASR inference over an audio file using Whisper model:
```python
from scripts.inference import infer

MODEL_NAME = 'medium'
AUDIO_FILE = 'example.wav'
OUTPUT_FILE = 'example.trs'
SPLIT_STEREO = True

LOAD = {"device": "cuda", "compute_type": "int8"}
w = infer(MODEL_NAME, load=LOAD)

TRANSCRIBE = {"beam_size": 5, "word_timestaps": false, "vad_filter": true}
with open(OUTPUT_FILE, 'w') as fdo:
     for l in w(AUDIO_FILE, split_stereo=SPLIT_STEREO, transcribe=TRANSCRIBE):
     	 fdo.write(l + '\n')
```

Or use:
```
$> python ./whisper-infer.py medium example.wav \
   --split_stereo \
   --output example.trs \
   --load '{"device": "cuda", "compute_type": "int8"}' \
   --transcribe '{"beam_size": 5, "word_timestaps": false, "vad_filter": true}'
```
`LOAD` is a dictionariy with key/values passed to faster_whisper [initialization](https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L584) function.
`TRANSCRIBE` is a dictionary with key/values passed to faster_whisper [transcribe](https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L705) function.

### eval-asr.py


### align-asr.py

