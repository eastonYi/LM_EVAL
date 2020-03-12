# coding=utf-8
import tokenization
import queue

MASKED_TOKEN = "[MASK]"


class TextDataSet():
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, data_file, vocab_file, max_seq_length):
        self.data_file = data_file
        # self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.size_dataset = self.get_size()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.MASKED_ID = self.tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]
        self.queue_tokens = queue.Queue(20000)
        self.queue_uttids = queue.Queue(2000)

    def __len__(self):
        return self.size_dataset

    def get_size(self):

        return sum(1 for line in open(self.data_file))

    def __iter__(self):
        with open(self.data_file) as f:
            for i, line in enumerate(f):
                uttid, text = line.strip().split()
                tokens = self.tokenizer.tokenize(text)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens) > self.max_seq_length - 2:
                    tokens = tokens[0:(self.max_seq_length - 2)]

                input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                len_pad = self.max_seq_length - len(input_tokens)
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
                input_mask = [1] * len(input_tokens) + [0] * len_pad

                [self.queue_tokens.put(i) for i in input_tokens]
                self.queue_uttids.put(uttid)

                if i % 1000 == 0:
                    print('processed {} sentences.'.format(i))

                yield from self.create_sequential_mask(input_tokens, input_ids, input_mask)

    def create_sequential_mask(self, input_tokens, input_ids, input_mask):
        """Mask each token/word sequentially"""

        for i in range(1, len(input_tokens)-1):
            mask_count = 1
            while is_subtoken(input_tokens[i+mask_count]):
                mask_count += 1

            input_ids_new, masked_lm_positions, masked_lm_labels = \
                self.create_masked_lm_prediction(input_ids, i, mask_count)
            pad_len = self.max_seq_length - len(masked_lm_positions)

            masked_lm_positions += [0] * pad_len
            masked_lm_labels += [0] * pad_len

            i += mask_count
            output = (input_ids_new, input_mask, masked_lm_positions, masked_lm_labels)

            yield output

    def create_masked_lm_prediction(self, input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        candidate_lm_labels = []
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID
            candidate_lm_labels.append(input_ids[i])

        return new_input_ids, masked_lm_positions, candidate_lm_labels


def is_subtoken(x):
    return x.startswith("##")


class ASRDecoded(TextDataSet):
    def __init__(self, data_file, ref_file, vocab_file, max_seq_length):
        self.ref_file = ref_file
        super().__init__(data_file, vocab_file, max_seq_length)

    def __iter__(self):
        with open(self.ref_file) as f_ref, open(self.data_file) as f:

            for i, (line_ref, line) in enumerate(zip(f_ref, f)):
                _, ref = line_ref.strip().split()
                _, text, candidates = line.strip().split(',', maxsplit=2)
                tokens = list(text)
                list_cands = candidates.split()
                assert len(list_cands) == len(tokens)

                list_decoded_cands = [] # [[*], [*], [*, *], [*], [*, *]]
                list_vague_idx = []
                list_vague_cands = []
                for i, cands in enumerate(list_cands):
                    list_cands = [] # [*] or [*, *]
                    for cand in cands.split(','):
                        token, prob = cand.split(':')
                        prob = float(prob)
                        if prob > 0.01:
                            list_cands.append(token)
                    if len(list_cands) > 1:
                        list_vague_idx.append(i)
                        list_vague_cands.append(self.tokenizer.convert_tokens_to_ids(list_cands))
                    list_decoded_cands.append(list_cands)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens) > self.max_seq_length - 2:
                    print("there is a sent's length larger than {}".format(self.max_seq_length))
                    tokens = tokens[0:(self.max_seq_length - 2)]

                input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                len_pad = self.max_seq_length - len(input_tokens)
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
                input_mask = [1] * len(input_tokens) + [0] * len_pad

                [self.queue_tokens.put(i) for i in list_decoded_cands]
                self.queue_uttids.put(ref)

                if i % 1000 == 0:
                    print('processed {} sentences.'.format(i))

                yield from self.create_sequential_mask(
                    input_tokens, input_ids, input_mask, list_vague_idx, list_vague_cands)

    def create_sequential_mask(self, input_tokens, input_ids, input_mask,
                               list_vague_idx, list_vague_cands):
        """Mask each token/word sequentially"""

        assert len(list_vague_idx) == len(list_vague_cands)
        list_vague_cands.reverse()
        for i in range(1, len(input_tokens)-1):

            if i-1 not in list_vague_idx: continue
            mask_count = 1
            while is_subtoken(input_tokens[i+mask_count]):
                mask_count += 1

            cand_ids = list_vague_cands.pop()
            input_ids_new, masked_lm_positions, masked_lm_labels = \
                self.create_masked_lm_prediction(input_ids, i, mask_count)

            masked_lm_positions += [0] * (self.max_seq_length - len(masked_lm_positions))
            cand_ids += [0] * (self.max_seq_length - len(cand_ids))

            i += mask_count
            output = (input_ids_new, input_mask, masked_lm_positions, masked_lm_labels)
            print(i)

            yield output

    def create_masked_lm_prediction(self, input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        candidate_lm_labels = []
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID
            candidate_lm_labels.append(input_ids[i])

        return new_input_ids, masked_lm_positions, candidate_lm_labels
