def gen_top1(args):
    with open(args.input) as f, open(args.output, 'w') as fw:
        for i, line in enumerate(f):
            if i % args.top == 0:
                fw.write(line.strip() + '\n')


def gen_vote(args):
    def vote(list_trans):
        list_voted = []
        for cands in zip(list_trans):
            list_voted.append(max(set(cands), key=cands.count))

        return list_voted

    with open(args.input) as f, open(args.output, 'w') as fw:
        list_trans = []
        list_uttid = []
        max_len = 0
        for i, line in enumerate(f):
            uttid, trans = line.strip().split(' ', 1)
            trans = trans.split()
            list_trans.append(trans)
            list_uttid.append(uttid)
            if max_len < len(trans): max_len = len(trans)

            if i % args.top == args.top - 1:
                list_trans_pad = []
                for trans in list_trans:
                    list_trans_pad.append(trans + ['']*(max_len-len(trans)))
                trans_voted = vote(list_trans_pad)
                assert len(set(list_uttid)) == 1
                fw.write(uttid + ' ' + ' '.join(i for i in trans_voted if i) + '\n')
                list_trans = []
                list_uttid = []
                max_len = 0


def gen_rerank(args):
    with open(args.input) as f, open(args.output, 'w') as fw:
        list_trans = []
        list_uttid = []
        list_score = []
        for i, line in enumerate(f):
            uttid, score, trans = line.strip().split(' ', 2)
            list_trans.append(trans)
            list_uttid.append(uttid)
            list_score.append(float(score))

            if i % args.top == args.top - 1:
                assert len(set(list_uttid)) == 1
                trans_voted = sorted(list_trans, key=list_score)[0]
                fw.write(uttid + ' ' + trans_voted + '\n')
                list_trans = []
                list_uttid = []
                list_score = []


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode')
    parser.add_argument('--input', type=str, dest='input')
    parser.add_argument('--output', type=str, dest='output')
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.0)
    parser.add_argument('--top', type=int, dest='top', default=5)
    args = parser.parse_args()

    if args.mode == 'vote':
        gen_vote(args)
    elif args.mode == 'top1':
        gen_top1(args)
    elif args.mode == 'rerank':
        gen_rerank(args)
