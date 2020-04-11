import editdistance as ed
from pypinyin import pinyin, lazy_pinyin, Style


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


def cand_cer_upper(cand_file, output_file, threshold):
    batch_res_dist = 0
    batch_fixed_dist = 0
    batch_len = 0
    num_not_equal = 0
    with open(cand_file) as f, open(output_file, 'w') as fw:
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
        if float(cand.split(':')[1]) > threshold:
            list_tokens.append(cand.split(':')[0])

    if len(list_tokens) > 1:
        list_pinyin = [pinyin(token)[0][0] for token in list_tokens]
        list_tokens = tokens_mode_by_pinyin(list_pinyin, list_tokens)

    if not list_tokens:
        list_tokens.append(list_cands[0].split(':')[0])

    return list_tokens


def tokens_mode_by_pinyin(list_pinyin, list_tokens):
    pinyin_mode = max(list_pinyin, key=list_pinyin.count)
    _list = []
    for p, token in zip(list_pinyin, list_tokens):
        if p == pinyin_mode:
            _list.append(token)
    return _list


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode')
    parser.add_argument('--input', type=str, dest='input')
    parser.add_argument('--output', type=str, dest='output')
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.0)
    args = parser.parse_args()

    if args.mode == 'cand':
        cand_cer_upper(args.input, args.output, args.threshold)
    elif args.mode == 'fixed':
        fixed_cer(args.input)
