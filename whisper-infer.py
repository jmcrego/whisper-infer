import os
import sys
import json
import logging
import argparse
from scripts import infer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to run inference on Whisper models. Ex: python whisper-infer.py small --audio example.wav --load '{\"device\": \"cuda\", \"compute_type\": \"int8\"}' --transcribe '{\"language\": \"en\", \"beam_size\": 5, \"word_timestamps\": true}'")
    parser.add_argument('model', type=str, help='Path or name of the Whisper model. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/utils.py#L12)')
    parser.add_argument('--audio', type=str, required=True, help='Audio file to transcribe.')    
    parser.add_argument('--channel', type=int, default=None, help='Transcribe channel of the audio (0 or 1).')    
    parser.add_argument('--start', type=float, default=None, help='Transcribe starting at this second.')    
    parser.add_argument('--end', type=float, default=None, help='Transcribe ending at this second.')    
    parser.add_argument('--output', type=str, default=None, required=False, help='File where to print transcriptions (or print on STDOUT).')
    parser.add_argument('--save', type=str, default=None, required=False, help='File where to save the audio segment.')
    parser.add_argument('--split_stereo', action='store_true', help='Split channels of (stereo) audio files')
    parser.add_argument('--load', type=str, default=None, required=False, help="JSON dictionary to control model loading. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L583)")
    parser.add_argument('--transcribe', type=str, default=None, required=False, help="JSON dictionary to control transcription. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L705)")
    parser.add_argument('--force', action='store_true', help='Rewrite output file if exists')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'WARNING' if not args.verbose else 'INFO'), filename=None)

    if args.output is not None and args.force==False and os.path.exists(args.output):
        raise ValueError(f"Ouput file {args.output} already exists! use --force to overwrite.")
    
    load = json.loads(args.load) if args.load is not None else {}
    transcribe = json.loads(args.transcribe) if args.transcribe is not None else {}
    wi = infer(args.model, args.audio, split_stereo=args.split_stereo, load=load)
    res = wi(channel=args.channel, start=args.start, end=args.end, transcribe=transcribe, save=args.save)    

    fdo = sys.stdout if args.output is None else open(args.output, 'w')

    for e in res:
        fdo.write(f"{e}\n")

    if args.output is not None:
        fdo.close()