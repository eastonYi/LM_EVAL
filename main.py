"""BERT language model predict."""
import numpy as np
import tensorflow as tf

import modeling

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string(
        "input_file", 'test.zh.tsv',
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")
flags.DEFINE_string(
        "ref_file", 'test.zh.tsv',
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")
flags.DEFINE_string(
        "output", None,
        "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
        "max_seq_length", 30,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")


def model_builder(bert_config, init_checkpoint, output_logits=False):
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

        if output_logits:
            inputs = (input_ids, input_mask, masked_lm_positions, masked_lm_positions)

            bacth_size, seq_len = tf.shape(masked_lm_positions)[0], tf.shape(masked_lm_positions)[1]
            outputs = tf.reshape(
                log_probs, [bacth_size, seq_len, bert_config.vocab_size])

        else:
            masked_lm_ids = tf.placeholder(tf.int32, [None, None], name='masked_lm_ids')
            inputs = (input_ids, input_mask, masked_lm_positions, masked_lm_positions)

            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            one_hot_labels = tf.one_hot(
                masked_lm_ids, depth=bert_config.vocab_size, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

            outputs = tf.reshape(per_example_loss, [-1, tf.shape(masked_lm_positions)[1]])

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


def sorting():
    from data_reader import TextDataSet

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

    dataset = TextDataSet(FLAGS.input_file, FLAGS.vocab_file, FLAGS.max_seq_length)

    batch_iter = tf.data.Dataset.from_generator(
        lambda: dataset,
        (tf.int32,) * 4,
        (tf.TensorShape([None]),) * 4).batch(8).make_initializable_iterator()

    prob_op = model_builder(batch_iter, bert_config, FLAGS.init_checkpoint)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        sess.run(batch_iter.initializer)
        try:
            with open(FLAGS.output, 'w') as fw:
                tf.logging.info("***** Predict results *****")
                tf.logging.info("Saving results to %s" % FLAGS.output)
                list_tokens = []
                list_scores = []
                while True:
                    probs = sess.run(prob_op)

                    for word_loss in probs:
                        # start of a sentence
                        token = dataset.queue_tokens.get()
                        if token == "[CLS]":
                            sentence_loss = 0.0
                            word_count_per_sent = 0
                            uttid = dataset.queue_uttids.get()
                            token = dataset.queue_tokens.get()
                        elif token == "[SEP]":
                            new_line = uttid + \
                                        'preds:{},'.format(' '.join(list_tokens)) + \
                                        'score_lm:{},'.format(' '.join(list_scores)) + \
                                        'ppl:{:.2f}'.format(float(np.exp(sentence_loss / word_count_per_sent)))
                            fw.write(new_line+'\n')
                            list_tokens = []
                            list_scores = []
                            token = dataset.queue_tokens.get()
                        # add token
                        list_tokens.append(token)
                        list_scores.append('{:.3f}'.format(np.exp(-word_loss[0])))
                        sentence_loss += word_loss[0]
                        word_count_per_sent += 1

        except tf.errors.OutOfRangeError:
            tf.logging.info("***** Finished *****")


def fixing():
    from data_reader import ASRDecoded

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("***** Running Fixing *****")
    tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

    dataset = ASRDecoded(FLAGS.input_file, FLAGS.ref_file, FLAGS.vocab_file, FLAGS.max_seq_length)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    input_pl, log_prob_op = model_builder(bert_config, FLAGS.init_checkpoint, output_logits=True)

    tf.logging.info("***** Predict results *****")
    tf.logging.info("Saving results to %s" % FLAGS.output)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        with open(FLAGS.output, 'w') as fw:
            for sent in dataset:
                ref = sent[0]
                list_decoded_cands = sent[1]
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
                    for cands in list_decoded_cands:
                        if len(cands) > 1:
                            list_cands = []
                            log_prob = next(iter_log_probs)
                            cands_ids = dataset.tokenizer.convert_tokens_to_ids(cands)
                            for cand, cand_id in zip(cands, cands_ids):
                                cand_id
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
                    print(new_line)
                fw.write(new_line+'\n')


if __name__ == "__main__":
    # main()
    fixing()
