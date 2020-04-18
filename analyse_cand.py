import editdistance as ed
import re


def fixed_cer(fixed_file):
    batch_res_dist = 0
    batch_fixed_dist = 0
    batch_len = 0
    num_not_equal = 0
    with open (fixed_file) as f:
        for i, line in enumerate(f):
            try:
                _, ref, res, fixed = line.strip().split(',', 3)
                ref = ref.split(':')[1].split()
                res = res.split(':')[1].split()
                fixed = fixed.split(':', 1)[1].split()
                if len(ref) != len(res):
                    num_not_equal += 1
                    continue
            except:
                print(line)
                continue
            batch_res_dist += ed.eval(res, ref)
            batch_fixed_dist += ed.eval(fixed, ref)
            batch_len += len(ref)

    cer_res = batch_res_dist / batch_len
    cer_fixed = batch_fixed_dist / batch_len

    print('res CER: {:.3f}; fixed CER: {:.3f}; not euqal:{}/{}'.format(
            cer_res, cer_fixed, num_not_equal, i))


def cand_cer_upper(cand_file, ref_fixed, threshold):
    batch_res_dist = 0
    batch_fixed_dist = 0
    batch_len = 0
    num_not_equal = 0
    with open(cand_file) as f, open(ref_fixed, 'w') as fw:
        for i, line in enumerate(f):
            uttid, ref, res, all_cands = line.strip().split(',', 3)
            uttid = uttid.split(':')[1]
            ref = ref.split(':')[1].split()
            res = res.split(':')[1].split()
            all_cands = all_cands.split(':', 1)[1].split()

            ref_fixed = []
            if len(ref) == len(res):
                for token, cands in zip(ref, all_cands):
                    cand_tokens = cand_filter(cands.split(','), threshold)
                    if token in cand_tokens:
                        ref_fixed.append(token)
                    else:
                        ref_fixed.append(cand_tokens[0])
                new_line = ' '.join(ref_fixed)

                fw.write(uttid + ' ' + new_line + '\n')

                batch_res_dist += ed.eval(res, ref)
                batch_fixed_dist += ed.eval(ref_fixed, ref)
                batch_len += len(ref)
            else:
                num_not_equal += 1

    cer_res = batch_res_dist / batch_len
    cer_fixed = batch_fixed_dist / batch_len

    print('res CER: {:.3f}; ref-fixed CER: {:.3f}; not euqal:{}/{}'.format(
           cer_res, cer_fixed, num_not_equal, i))


def cand_filter(list_cands, threshold=0.0):
    list_tokens= []

    for cand in list_cands:
        token, p = cand.split(':')
        if len(token)==1 and float(p) > threshold:
            list_tokens.append(token)

    if not list_tokens:
        list_tokens.append(list_cands[0].split(':')[0])

    return list_tokens


def fixed2trans(f_fix, f_fixed_trans):
    i = True
    num_saved = 0
    with open(f_fix) as f, open(f_fixed_trans, 'w') as fw:
        for j, line in enumerate(f):
            uttid, ref, res, fixed = line.strip().split(',', 3)
            uttid = uttid.split(':')[1]
            ref = ref.split(':')[1]
            res = res.split(':')[1]
            fixed = fixed.split(':', 1)[1]

            if i:
                print(ref, res, fixed)
                print(len(ref), len(res))
                i = False

            if len(ref) - 1 < len(res) < len(ref) + 1:
                fw.write(uttid + ' ' + ' '.join(fixed) + '\n')
                num_saved += 1
    print('input: {}, output: {}'.format(j, num_saved))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode')
    parser.add_argument('--input', type=str, dest='input')
    parser.add_argument('--output', type=str, dest='output')
    parser.add_argument('--output2', type=str, dest='output2')
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.0)
    args = parser.parse_args()

    if args.mode == 'cand':
        cand_cer_upper(args.input, args.output, args.output2, args.threshold)
    elif args.mode == 'fixed':
        fixed_cer(args.input)
    elif args.mode == 'fixed2trans':
        fixed2trans(args.input, args.output)
