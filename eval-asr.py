import sys
import os
import argparse
from scripts.utils import jiwer_wrap, hyp2list, ref2list, align_hyp_to_ref

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to evaluate ASR transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hyps', type=str, required=True, help='File with raw hypotheses.')
    parser.add_argument('--refs', type=str, required=True, help='File with raw references.')
    parser.add_argument('--channel', type=int, default=1, help='Channel to use when stereo audio.')
    parser.add_argument('--measures', action='store_true', help='show measures')
    parser.add_argument('--alignments', action='store_true', help='show alignments')
    parser.add_argument('--uppercase', action='store_true', help='uppercase refs/hyps')
    parser.add_argument('--no_punct', action='store_true', help='delete punctuation from refs/hyps')
    parser.add_argument('--no_hesit', action='store_true', help='delete hesitations from refs/hyps')
    parser.add_argument('--no_noise', action='store_true', help='delete noise marks from refs/hyps')
    parser.add_argument('--split_apos', action='store_true', help='split apostrophes')
    parser.add_argument('--single_line', action='store_true', help='merged refs/hyps in a single line')
    args = parser.parse_args()

    #refs = ref2list(args.refs, args.channel)
    #hyps = hyp2list(args.hyps, args.channel)
    h2r = align_hyp_to_ref()
    hyps, refs = h2r(args.hyps, args.refs, args.channel)
    wer = jiwer_wrap(show_measures=args.measures, show_alignments=args.alignments, uppercase=args.uppercase, no_punct=args.no_punct, no_hesit=args.no_hesit, no_noise=args.no_noise, split_apos=args.split_apos, single_line=args.single_line)
    wer(hyps, refs)



