import tokenization
import argparse
import codecs


def tokenizer_file(vocab, fin, fout):
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab, do_lower_case=True)

    writer = codecs.open(fout, "w", 'utf8')
    with codecs.open(fin, "r", 'utf8') as reader:
       for line in reader:
            line = tokenization.convert_to_unicode(line)
            line = line.strip()
            line_tokens = tokenizer.tokenize(line)
            print(' '.join(line_tokens), file=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--fin', type=str)
    parser.add_argument('--fout', type=str)

    args = parser.parse_args()
    tokenizer_file(args.vocab, args.fin, args.fout)
