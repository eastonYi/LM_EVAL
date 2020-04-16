"""BERT language model predict."""
import numpy as np
import tensorflow as tf

import modeling

predict_batch_size = 8


def model_builder(bert_config, init_checkpoint):
    """The `model_fn` for TPUEstimator."""
    input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
    masked_lm_positions = tf.placeholder(tf.int32, [None, None], name='masked_lm_positions')
    segment_ids = tf.zeros_like(input_mask)

    with tf.device("/gpu:0"):
        model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=(input_mask),
                token_type_ids=segment_ids)

        log_probs = get_masked_lm_output(
                bert_config,
                model.get_sequence_output(),
                model.get_embedding_table(),
                masked_lm_positions)

        inputs = (input_ids, input_mask, masked_lm_positions, masked_lm_positions)

        bacth_size, seq_len = tf.shape(masked_lm_positions)[0], tf.shape(masked_lm_positions)[1]
        outputs = tf.reshape(
            log_probs, [bacth_size, seq_len, bert_config.vocab_size])

    tvars = tf.trainable_variables()
    initialized_variable_names = {}

    if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    return inputs, outputs


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

    return log_probs


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def load_bert_model(bert_dir):
    bert_config = bert_dir+'/bert_config.json'
    bert_model = bert_dir + '/bert_model.ckpt'
    vocab = bert_dir + '/vocab.txt'

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(bert_config)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("    Batch size = %d", predict_batch_size)

    input_pl, log_prob_op = model_builder(bert_config, bert_model)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    return vocab, input_pl, log_prob_op, config


def rerank(args):
    from data_reader import TextDataSet

    dataset = TextDataSet(args.input, args.bert_dir + '/vocab.txt', args.max_seq_length)

    vocab, input_pl, log_prob_op, config = load_bert_model(args.bert_dir)

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        with open(args.output, 'w') as fw:
            for uttid, sent, sent_inputs in dataset:
                list_scores = []

                inputs = [np.array(i, dtype=np.int32) for i in zip(*sent_inputs)]
                dict_feed = {input_pl[0]: inputs[0],
                             input_pl[1]: inputs[1],
                             input_pl[2]: inputs[2]}
                log_probs = sess.run(log_prob_op, feed_dict=dict_feed)

                assert len(sent) == len(log_probs)
                token_ids = dataset.tokenizer.convert_tokens_to_ids(sent)
                for token_id, log_prob in zip(token_ids, log_probs):
                    # start of a sentence
                    list_scores.append(np.exp(log_prob[0][token_id]))

                ppl = np.mean(list_scores)
                new_line = 'id:' + uttid + \
                            ',preds:{}'.format(' '.join(sent)) + \
                            ',score_lm:{}'.format(' '.join('{:.3f}'.format(socre) for socre in list_scores)) + \
                            ',ppl:{:.2f}'.format(ppl)
                fw.write(new_line+'\n')

            tf.logging.info("***** Finished *****")


def fix(args):
    from data_reader import ASRDecoded

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("***** Running Fixing *****")

    vocab, input_pl, log_prob_op, config = load_bert_model(args.bert_dir)

    dataset = ASRDecoded(args.input, args.ref_file, args.vocab, args.max_seq_length)

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        with open(args.output, 'w') as fw:
            for sent in dataset:
                ref = sent[0]
                list_decoded_cands = sent[1]
                print(list_decoded_cands)
                if len(sent) == 2:
                    str_org  = ''.join(i[0] for i in list_decoded_cands)
                    new_line = 'ref:' + ref + ',asr output:' + str_org
                elif len(sent) == 3:
                    list_vagues_inputs = sent[2]
                    list_fixed = []
                    list_orgin = []
                    vague_inputs = [np.array(i, dtype=np.int32) for i in zip(*list_vagues_inputs)]
                    dict_feed = {input_pl[0]: vague_inputs[0],
                                 input_pl[1]: vague_inputs[1],
                                 input_pl[2]: vague_inputs[2]}
                    log_probs = sess.run(log_prob_op, feed_dict=dict_feed)
                    iter_log_probs = iter(log_probs)
                    assert sum([1 for i in list_decoded_cands if len(i)>1]) == len(log_probs)
                    for cands in list_decoded_cands:
                        if len(cands) > 1:
                            list_cands = []
                            log_prob = next(iter_log_probs)
                            cands_ids = dataset.tokenizer.convert_tokens_to_ids(cands)
                            for cand, cand_id in zip(cands, cands_ids):
                                list_cands.append('{}:{:.2e}'.format(cand, np.exp(log_prob[0][cand_id])))
                            list_cands.sort(key=lambda x: float(x.split(':')[1]), reverse=True)
                            list_fixed.append('(')
                            list_fixed.append(','.join(list_cands))
                            list_fixed.append(')')
                        else:
                            list_fixed.append(cands[0])
                        list_orgin.append(cands[0])
                    str_org = ''.join(list_orgin)
                    str_fixed = ''.join(list_fixed)
                    new_line = 'ref:' + ref + ',asr output:' + str_org + ',lm fixed:' + str_fixed
                fw.write(new_line+'\n')


def iter_fix():
    from data_reader import cand_filter, choose, ASRDecoded_iter

    vocab, input_pl, log_prob_op, config = load_bert_model(args.bert_dir)

    dataset = ASRDecoded_iter(args.input, args.vocab, args.max_seq_length, args.is_cn)

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        with open(args.output, 'w') as fw:
            for sent in dataset:
                uttid, ref, res, list_all_cands = sent
                list_all_cands, list_vague_idx = cand_filter(
                    list_all_cands, threshold=args.threshold, is_cn=args.is_cn)

                while list_vague_idx:
                    list_vague_idx = choose(list_all_cands)
                    if list_vague_idx:
                        list_vagues_inputs = dataset.gen_input(list_all_cands, list_vague_idx)
                        vague_inputs = [np.array(i, dtype=np.int32) for i in zip(*list_vagues_inputs)]
                        dict_feed = {input_pl[0]: vague_inputs[0],
                                     input_pl[1]: vague_inputs[1],
                                     input_pl[2]: vague_inputs[2]}
                        log_probs = sess.run(log_prob_op, feed_dict=dict_feed)
                        assert len(log_probs) == len(list_vague_idx)
                        iter_log_probs = iter(log_probs)
                        for i in list_vague_idx:
                            cands = list_all_cands[i]
                            list_tokens = [i.split(':')[0] for i in cands]
                            log_prob = next(iter_log_probs)
                            cands_ids = dataset.tokenizer.convert_tokens_to_ids(list_tokens)
                            list_cands = []
                            for cand, cand_id in zip(cands, cands_ids):
                                list_cands.append((cand, np.exp(log_prob[0][cand_id])))
                            list_cands.sort(key=lambda x: x[1], reverse=True)
                            list_all_cands[i] = list_cands[0][0][0]
                try:
                    while list_vague_idx:
                        list_vague_idx = choose(list_all_cands)
                        if list_vague_idx:
                            list_vagues_inputs = dataset.gen_input(list_all_cands, list_vague_idx)
                            vague_inputs = [np.array(i, dtype=np.int32) for i in zip(*list_vagues_inputs)]
                            dict_feed = {input_pl[0]: vague_inputs[0],
                                         input_pl[1]: vague_inputs[1],
                                         input_pl[2]: vague_inputs[2]}
                            log_probs = sess.run(log_prob_op, feed_dict=dict_feed)
                            assert len(log_probs) == len(list_vague_idx)
                            iter_log_probs = iter(log_probs)
                            for i in list_vague_idx:
                                cands = list_all_cands[i]
                                list_tokens = [i.split(':')[0] for i in cands]
                                log_prob = next(iter_log_probs)
                                cands_ids = dataset.tokenizer.convert_tokens_to_ids(list_tokens)
                                list_cands = []
                                for cand, cand_id in zip(cands, cands_ids):
                                    list_cands.append((cand, np.exp(log_prob[0][cand_id])))
                                list_cands.sort(key=lambda x: x[1], reverse=True)
                                list_all_cands[i] = list_cands[0][0][0]

                except KeyError:
                    print(res, list_all_cands, 'cand OOV')
                    list_all_cands = []

                for cands in list_all_cands:
                    if len(cands) > 1:
                        list_all_cands = []
                        break
                if not list_all_cands:
                    continue

                fixed = ' '.join(list_all_cands)
                new_line = 'uttid:{},ref:{},res:{},fixed:{}'.format(uttid, ref, res, fixed)
                fw.write(new_line+'\n')


if __name__ == "__main__":
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode')
    parser.add_argument('--input', type=str, dest='input_file')
    parser.add_argument('--output', type=str, dest='new_file')
    parser.add_argument('--bert_dir', type=str, dest='bert_dir')
    parser.add_argument('--trans', type=str, dest='trans')
    parser.add_argument('--max_seq_length', type=int, dest='max_seq_length', default=30)
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.002)
    parser.add_argument('--is_cn', action='store_true', default=False)
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    tf.logging.info("Saving results to %s" % args.new_file)

    if args.mode == 'rerank':
        rerank(args)
    elif args.mode == 'fix':
        fix(args)
    elif args.mode == 'iter_fix':
        iter_fix(args)

    logging.info("Done")
