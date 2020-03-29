# coding=utf-8
import tokenization
import re

MASKED_TOKEN = "[MASK]"


class TextDataSet():
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, data_file, vocab_file, max_seq_length):
        self.data_file = data_file
        self.size_dataset = self.get_size()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.MASKED_ID = self.tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]

    def __len__(self):
        return self.size_dataset

    def get_size(self):

        return sum(1 for line in open(self.data_file))

    def __iter__(self):
        with open(self.data_file) as f:
            for i, line in enumerate(f):
                uttid, text = line.strip().split()
                tokens = self.tokenizer.tokenize(text)

                try:
                    assert len(text) <= self.max_seq_length - 2
                except AssertionError:
                    print(text, '. too long')
                    continue

                try:
                    input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                    len_pad = self.max_seq_length - len(input_tokens)
                    input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
                    input_mask = [1] * len(input_tokens) + [0] * len_pad
                except KeyError:
                    print(text, '. OOV')
                    continue

                if i % 1000 == 0:
                    print('processed {} sentences.'.format(i))

                yield uttid, tokens, self.create_sequential_mask(input_tokens, input_ids, input_mask)

    def create_sequential_mask(self, input_tokens, input_ids, input_mask):
        """Mask each token/word sequentially"""

        list_inputs = []
        for i in range(1, len(input_tokens)-1):
            mask_count = 1
            while is_subtoken(input_tokens[i+mask_count]):
                mask_count += 1

            input_ids_new, masked_lm_positions = \
                self.create_masked_lm_prediction(input_ids, i, mask_count)
            pad_len = self.max_seq_length - len(masked_lm_positions)

            masked_lm_positions += [0] * pad_len

            i += mask_count
            output = (input_ids_new, input_mask, masked_lm_positions)
            list_inputs.append(output)

        return list_inputs

    def create_masked_lm_prediction(self, input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        candidate_lm_labels = []
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID
            candidate_lm_labels.append(input_ids[i])

        return new_input_ids, masked_lm_positions



def is_subtoken(x):
    return x.startswith("##")


class ASRDecoded(TextDataSet):
    def __init__(self, data_file, ref_file, vocab_file, max_seq_length):
        self.ref_file = ref_file
        super().__init__(data_file, vocab_file, max_seq_length)

    def __iter__(self):
        with open(self.ref_file) as f_ref, open(self.data_file) as f, open('samples.droped', 'w') as fw:
            num_converted = 0
            for i, (line_ref, line) in enumerate(zip(f_ref, f)):
                uttid, ref = line_ref.strip().split()
                _uttid, text, candidates = line.strip().split(',', maxsplit=2)

                assert uttid == _uttid

                # check if the sent is too long or contains OOV
                if not self.check(text, ref, fw):
                    continue

                tokens = list(text)
                list_all_cands = candidates.split()
                assert len(list_all_cands) == len(tokens)

                list_decoded_cands = [] # [[*], [*], [*, *], [*], [*, *]]
                list_vague_idx = []
                list_vague_cands = []
                try:
                    self.tokenizer.convert_tokens_to_ids(tokens)
                    for j, cands in enumerate(list_all_cands):
                        list_cands = [] # [*] or [*, *]
                        for cand in cands.split(','):
                            token, prob = cand.split(':')
                            prob = float(prob)
                            if prob > 0.01:
                                list_cands.append(token)
                        if len(list_cands) > 1:
                            cands_id = self.tokenizer.convert_tokens_to_ids(list_cands)
                            list_vague_cands.append(cands_id)
                            list_vague_idx.append(j)
                        elif len(list_cands) == 0:
                            list_cands.append(token)
                        list_decoded_cands.append(list_cands)
                except KeyError:
                    fw.write(ref + ', ' + text + ' OOV \n')
                    continue

                input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                len_pad = self.max_seq_length - len(input_tokens)
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
                input_mask = [1] * len(input_tokens) + [0] * len_pad

                num_converted += 1

                if i % 1000 == 0:
                    print('processed {} sentences.'.format(i))
                # print(list_decoded_cands)
                if list_vague_idx:
                    tmp = self.create_sequential_mask(input_ids, input_mask, list_vague_idx)
                    yield uttid, ref, list_decoded_cands, tmp
                else:
                    yield uttid, ref, list_decoded_cands

        print('***************utilized {}/{} samples to be fake samples*********************'.format(num_converted, i+1))

    def create_sequential_mask(self, input_ids, input_mask, list_vague_idx):
        """Mask each token/word sequentially"""
        list_outputs = []
        for i in list_vague_idx:
            input_ids_new, masked_lm_positions = \
                self.create_masked_lm_prediction(input_ids, i+1)

            masked_lm_positions += [0] * (self.max_seq_length - len(masked_lm_positions))
            output = (input_ids_new, input_mask, masked_lm_positions)
            list_outputs.append(output)

        return list_outputs

    def create_masked_lm_prediction(self, input_ids, mask_position):
        new_input_ids = list(input_ids)
        masked_lm_positions = list(range(mask_position, mask_position+1))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID

        return new_input_ids, masked_lm_positions

    def check(self, text, ref, fw):
        # filter samples
        try:
            assert re.findall("[\u4e00-\u9fa5]+", text)[0] == text
            assert len(self.tokenizer.tokenize(ref)) == len(ref)
        except (IndexError, AssertionError):
            fw.write(ref + ', ' + text + ' subword' + '\n')
            return 0

        try:
            assert len(text) <= self.max_seq_length - 2
        except AssertionError:
            fw.write(ref + ', ' + text + ' too long\n')
            return 0

        return 1


class ASRDecoded2(ASRDecoded):
    def __init__(self, data_file, vocab_file, max_seq_length):
        super().__init__(data_file, None, vocab_file, max_seq_length)

    def __iter__(self):
        with open(self.data_file) as f, open('samples.droped', 'w') as fw:
            num_converted = 0
            for i, line in enumerate(f):
                uttid, ref, res, candidates = \
                    [i.split(':', maxsplit=1)[1] for i in line.strip().split(',', maxsplit=3)]

                # filter samples
                if not self.check(res, fw):
                    continue

                tokens = list(res)
                list_all_cands = candidates.split()
                assert len(list_all_cands) == len(tokens)

                num_converted += 1

                if i % 1000 == 0:
                    print('processed {} sentences.'.format(i))

                yield uttid, ref, res, list_all_cands

        print('***************utilized {}/{} samples to be fake samples*********************'.format(num_converted, i+1))

    def gen_input(self, list_decoded_cands, list_vague_idx):
        assert list_vague_idx
        list_res_new = [i[0].split(':')[0] for i in list_decoded_cands]
        input_tokens = ["[CLS]"] + list_res_new + ["[SEP]"]
        len_pad = self.max_seq_length - len(input_tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
        input_mask = [1] * len(input_tokens) + [0] * len_pad

        return self.create_sequential_mask(input_ids, input_mask, list_vague_idx)

    def check(self, text, fw):
        # filter samples
        try:
            assert re.findall("[\u4e00-\u9fa5]+", text)[0] == text
        except (IndexError, AssertionError):
            fw.write(text + ' subword' + '\n')
            return 0

        try:
            assert len(text) <= self.max_seq_length - 2
        except AssertionError:
            fw.write(text + ' too long\n')
            return 0

        return 1


def cand_threshold(list_all_cands):
    list_all_cands_new = []
    list_vague_idx = []
    for i, cands in enumerate(list_all_cands):
        list_cands = []
        for cand in cands.split(','):
            if float(cand.split(':')[1]) > 0.0:
                list_cands.append(cand)
        if len(list_cands) == 0:
            list_cands = cands.split(',')
            list_vague_idx.append(i)
        elif len(list_cands) == 1:
            list_cands = list_cands[0].split(':')[0]
        else:
            list_vague_idx.append(i)
        list_all_cands_new.append(list_cands)

    return list_all_cands_new, list_vague_idx


def choose(list_all_cands):
    list_idx = []
    anchors = []
    list_to_fix = []
    for i, x in enumerate(list_all_cands):
        if len(x) > 1:
            list_idx.append(i)
        else:
            anchors.append(i)
    for idx in list_idx:
        left_cond = (idx - 1 in anchors) and (idx - 2 in anchors)
        right_cond = (idx + 1 in anchors) and (idx + 2 in anchors)
        if left_cond or right_cond:
            list_to_fix.append(idx)

    return list_to_fix
