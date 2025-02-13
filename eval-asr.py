import argparse
from scripts.utils import jiwer_wrap, file2list

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to evaluate ASR transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hyp', type=str, required=True, help='File with raw hypotheses (one line each).')
    parser.add_argument('--ref', type=str, required=True, help='File with raw references (one line each).')
    parser.add_argument('--alignments', action='store_true', help='show alignments')
    parser.add_argument('--uppercase', action='store_true', help='uppercase refs/hyps')
    parser.add_argument('--no_punct', action='store_true', help='delete punctuation from refs/hyps')
    parser.add_argument('--no_hesit', action='store_true', help='delete hesitations from refs/hyps')
    parser.add_argument('--no_noise', action='store_true', help='delete noise marks from refs/hyps')
    parser.add_argument('--split_apos', action='store_true', help='split apostrophes')
    parser.add_argument('--single_line', action='store_true', help='merged refs/hyps in a single line')
    args = parser.parse_args()

    wer = jiwer_wrap(
        show_alignments=args.alignments, 
        uppercase=args.uppercase, 
        no_punct=args.no_punct, 
        no_hesit=args.no_hesit, 
        no_noise=args.no_noise, 
        split_apos=args.split_apos, 
        single_line=args.single_line,
    )
    print(wer(file2list(args.hyp), file2list(args.ref)))



