# coding=utf-8
import tokenization

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
        self.batch_tokens_tmp = []

    def __len__(self):
        return self.size_dataset

    def get_size(self):

        return sum(1 for line in open(self.data_file))

    def __iter__(self):
        """
        (Pdb) i
        [1,18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4,1]
        """
        with open(self.data_file) as f:
            for line in f:
                text = line.strip().split(',')[1]
                tokens = self.tokenizer.tokenize(text)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens) > self.max_seq_length - 2:
                    tokens = tokens[0:(self.max_seq_length - 2)]

                input_tokens = []
                segment_ids = []
                input_tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens:
                    input_tokens.append(token)
                    segment_ids.append(0)
                input_tokens.append("[SEP]")
                segment_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                self.batch_tokens_tmp.append(tokens)

                yield self.create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids)

    def create_sequential_mask(self, input_tokens, input_ids, input_mask, segment_ids):
        """Mask each token/word sequentially"""
        i = 1
        while i < len(input_tokens) - 1:
            mask_count = 1
            while is_subtoken(input_tokens[i+mask_count]):
                mask_count += 1

            input_ids_new, masked_lm_positions, masked_lm_labels = self.create_masked_lm_prediction(input_ids, i, mask_count)
            while len(masked_lm_positions) < self.max_seq_length:
                masked_lm_positions.append(0)
                masked_lm_labels.append(0)

            i += mask_count

        return input_ids_new, input_mask, segment_ids, masked_lm_positions, masked_lm_labels

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
