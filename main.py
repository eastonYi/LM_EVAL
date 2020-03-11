"""BERT language model predict."""
import numpy as np
import tensorflow as tf

import modeling
import tokenization
from data_reader import TextDataSet

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


def model_builder(features, bert_config, init_checkpoint):

    """The `model_fn` for TPUEstimator."""

    input_ids, input_mask, masked_lm_positions, masked_lm_ids = features.get_next()
    segment_ids = tf.zeros_like(input_mask)

    with tf.device("/gpu:0"):
        model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=(input_mask),
                token_type_ids=segment_ids)

        masked_lm_example_loss = get_masked_lm_output(
                bert_config,
                model.get_sequence_output(),
                model.get_embedding_table(),
                masked_lm_positions,
                masked_lm_ids)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}

    if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    return masked_lm_example_loss


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids):
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

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])

        one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
        # TODO: dynamic gather from per_example_loss
    return loss


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


def choose_device(op, device, default_device):
    if op.type.startswith('Variable'):
        device = default_device
    return device


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

    dataset = TextDataSet(FLAGS.input_file, FLAGS.vocab_file, FLAGS.max_seq_length)

    batch_iter = tf.data.Dataset.from_generator(
        lambda: dataset,
        (tf.int32,) * 4,
        (tf.TensorShape([None]),) * 4).cache().batch(FLAGS.predict_batch_size).prefetch(1).make_initializable_iterator()

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
                                        ',preds:{},'.format(' '.join(list_tokens)) + \
                                        'score_lm:{},'.format(' '.join(list_scores)) + \
                                        'ppl:{:.2f}'.format(float(np.exp(sentence_loss / word_count_per_sent)))
                            fw.write(new_line+'\n')
                            list_tokens = []
                            list_scores = []
                            token = dataset.queue_tokens.get()
                            sentence_loss = 0.0
                            word_count_per_sent = 0
                            uttid = dataset.queue_uttids.get()
                            token = dataset.queue_tokens.get()
                            
                        list_tokens.append(token)
                        list_scores.append('{:.3f}'.format(np.exp(-word_loss[0])))
                        sentence_loss += word_loss[0]
                        word_count_per_sent += 1

        except tf.errors.OutOfRangeError:
            tf.logging.info("***** Finished *****")


if __name__ == "__main__":
    tf.app.run()
