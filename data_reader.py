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
        self.queue_tokens = queue.Queue(100)

    def __len__(self):
        return self.size_dataset

    def get_size(self):

        return sum(1 for line in open(self.data_file))

    def __iter__(self):
        with open(self.data_file) as f:
            for line in f:
                text = line.strip().split(',')[1]
                tokens = self.tokenizer.tokenize(text)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens) > self.max_seq_length - 2:
                    tokens = tokens[0:(self.max_seq_length - 2)]

                input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
                segment_ids = [0] * self.max_seq_length
                len_pad = self.max_seq_length - len(input_tokens)
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * len_pad
                input_mask = [1] * len(input_tokens) + [0] * len_pad

                [self.queue_tokens.put(i) for i in input_tokens]

                yield from self.create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids)

    def create_sequential_mask(self, input_tokens, input_ids, input_mask, segment_ids):
        """Mask each token/word sequentially"""
        i = 1
        while i < len(input_tokens) - 1:
            mask_count = 1
            while is_subtoken(input_tokens[i+mask_count]):
                mask_count += 1

            input_ids_new, masked_lm_positions, masked_lm_labels = \
                self.create_masked_lm_prediction(input_ids, i, mask_count)
            pad_len = self.max_seq_length - len(masked_lm_positions)

            masked_lm_positions += [0] * pad_len
            masked_lm_labels += [0] * pad_len

            i += mask_count

            output = {"input_ids": input_ids_new,
                      "input_mask": input_mask,
                      "segment_ids": segment_ids,
                      "masked_lm_positions": masked_lm_positions,
                      "masked_lm_ids": masked_lm_labels}

            yield output

    def create_masked_lm_prediction(self, input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        masked_lm_labels = []
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASKED_ID
            masked_lm_labels.append(input_ids[i])
        return new_input_ids, masked_lm_positions, masked_lm_labels


def is_subtoken(x):
    return x.startswith("##")
