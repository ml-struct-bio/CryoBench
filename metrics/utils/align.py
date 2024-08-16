import argparse, os
from chimerax.core.commands import run

parser = argparse.ArgumentParser(description="Aligns two volumes")
parser.add_argument('ref', help='Input volume to align on')
parser.add_argument('vol', help='Input volume to align')
parser.add_argument('-o', type=os.path.abspath, required=True, help='Aligned mrc')
parser.add_argument('-f', type=os.path.abspath, required=True, \
    help='Text file that this program\'s output is being piped to (required if flip=True)')
parser.add_argument('--ninits', type=int, default=50, help='Number of alignments to try')
parser.add_argument('--flip', action='store_true', \
    help='Run an additional ninits alignments after flipping handedness of vol')
args = parser.parse_args()

run(session, 'open {}'.format(args.ref))
run(session, 'open {}'.format(args.vol))

if not args.flip:
    run(session, 'fitmap #2 inMap #1 search {}'.format(args.ninits))
    run(session, 'volume resample #2 onGrid #1 modelId #3')
    run(session, 'save {} #3'.format(args.o))
else:
    run(session, 'volume flip #2')
    run(session, 'fitmap #2 inMap #1 search {}'.format(args.ninits))
    run(session, 'fitmap #3 inMap #1 search {}'.format(args.ninits))

    corrs = []
    f = open(args.f, "r")
    for line in f:
        if line.startswith("  correlation"):
            corrs.append(float(line.split(",")[0][16:]))
    f.close()

    print(corrs)
    if corrs[0] > corrs[1]:
        run(session, 'volume resample #2 onGrid #1 modelId #4')
    else:
        run(session, 'volume resample #3 onGrid #1 modelId #4')
    run(session, 'save {} #4'.format(args.o))

run(session, 'exit')