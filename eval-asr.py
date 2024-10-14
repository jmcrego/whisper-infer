import sys
import os
import argparse
from whisper.utils import jiwer_wrap, file2list

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to evaluate ASR transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hyps', type=str, required=True, default='File with hypotheses.')
    parser.add_argument('--refs', type=str, required=True, default='File with references.')
    parser.add_argument('--measures', action='store_true', help='show measures')
    parser.add_argument('--alignments', action='store_true', help='show alignments')
    parser.add_argument('--uppercase', action='store_true', help='uppercase refs/hyps')
    parser.add_argument('--no_punct', action='store_true', help='delete punctuation from refs/hyps')
    parser.add_argument('--no_hesit', action='store_true', help='delete hesitations from refs/hyps')
    parser.add_argument('--no_noise', action='store_true', help='delete noise marks from refs/hyps')
    parser.add_argument('--split_apos', action='store_true', help='split apostrophes')
    args = parser.parse_args()

    refs = file2list(args.refs)
    hyps = file2list(args.hyps)
    wer = jiwer_wrap(show_measures=args.measures, show_alignments=args.alignments, uppercase=args.uppercase, no_punct=args.no_punct, no_hesit=args.no_hesit, no_noise=args.no_noise, split_apos=args.split_apos)
    wer(hyps, refs)



