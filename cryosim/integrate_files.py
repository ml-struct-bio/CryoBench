'''
Create a text file that integrate all files
'''

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='+', type=os.path.abspath, help='input files (list) to be combined')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='path to save integrated file')
    return parser

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def main(args):
    mkbasedir(args.o)
    mrcs_lst = [file for file in args.input if file.endswith('mrcs')]
    print(len(mrcs_lst))

    with open(args.o, 'w') as args.o:
        for item in mrcs_lst:
            args.o.write(item + '\n')

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)
