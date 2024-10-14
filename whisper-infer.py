import sys
import json
import logging
import argparse
from whisper.infer import infer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to run inference on Whisper models. Ex: python whisper-infer.py small --audio example.wav --load '{\"device\": \"cuda\", \"compute_type\": \"int8\"}' --transcribe '{\"language\": \"en\", \"beam_size\": 5, \"word_timestamps\": true}'")
    parser.add_argument('model', type=str, help='Path or name of the Whisper model. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/utils.py#L12)')
    parser.add_argument('--audio', type=str, required=True, help='Audio file to transcribe.')    
    parser.add_argument('--output', type=str, default=None, required=False, help='File where transcriptions are output (use stdout to print on STDOUT).')
    parser.add_argument('--split_stereo', action='store_true', help='Split channels of (stereo) audio files')
    parser.add_argument('--load', type=str, default=None, required=False, help="JSON dictionary to control model loading. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L583)")
    parser.add_argument('--transcribe', type=str, default=None, required=False, help="JSON dictionary to control transcription. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L705)")
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'WARNING' if not args.debug else 'INFO'), filename=None)
    
    load = json.loads(args.load) if args.load is not None else {}
    w = infer(args.model, load=load)

    if args.output is None:
        fdo = None
    elif args.output.lower() == 'stdout':
        fdo = sys.stdout
    else:
        fdo = open(args.output, 'w')
    transcribe = json.loads(args.transcribe) if args.transcribe is not None else {}
    res = w(args.audio, fdo=fdo, split_stereo=args.split_stereo, transcribe=transcribe)

    
