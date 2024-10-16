import os
import sys
import json
import logging
import argparse
from scripts.infer import infer
from scripts.utils import ref2list, jiwer_wrap
from faster_whisper import WhisperModel, decode_audio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to run inference on Whisper models. Ex: python whisper-infer.py small --audio example.wav --load '{\"device\": \"cuda\", \"compute_type\": \"int8\"}' --transcribe '{\"language\": \"en\", \"beam_size\": 5, \"word_timestamps\": true}'")
    parser.add_argument('model', type=str, help='Path or name of the Whisper model. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/utils.py#L12)')
    parser.add_argument('--audio', type=str, required=True, help='Audio file to transcribe.')    
    parser.add_argument('--reference', type=str, default=None, help='Reference transcription with segmentation timestamps.')    
    parser.add_argument('--load', type=str, default=None, required=False, help="JSON dictionary to control model loading. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L583)")
    parser.add_argument('--transcribe', type=str, default=None, required=False, help="JSON dictionary to control transcription. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L705)")
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'WARNING' if args.silent else 'INFO'), filename=None)

    load = json.loads(args.load) if args.load is not None else {}
    transcribe = json.loads(args.transcribe) if args.transcribe is not None else {}

    w = infer(args.model, load=load)

    refs = []
    hyps = []
    audio = decode_audio(args.audio, split_stereo=True)
    for ch in range(1, 3):
        ref = ref2list(args.reference, channel=ch)
        for e in ref:
            print(f"{e['beg']}, {e['end']}, {e['txt']}")
            w(audio[e['beg']:e['end']], split_stereo=False, transcribe=transcribe)
            refs.append(e['txt'])
            hyps.append(''.join([l for l in w]))
                        
    s = jiwer_wrap(uppercase=True, no_punct=True, no_hesit=False, no_noise=True, split_apos=True)
    s(refs, hyps)