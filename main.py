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

flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")


def model_fn_builder(bert_config, init_checkpoint):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]

        model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids)

        masked_lm_example_loss = get_masked_lm_output(
                bert_config,
                model.get_sequence_output(),
                model.get_embedding_table(),
                masked_lm_positions,
                masked_lm_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=masked_lm_example_loss, scaffold_fn=scaffold_fn)    # 输出mask_word的score
        return output_spec

    return model_fn


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


def score(result, queue_tokens, output_file):
    with open(output_file, 'w') as fw:
        tf.logging.info("***** Predict results *****")
        tf.logging.info("Saving results to %s" % FLAGS.output)
        list_tokens = []
        list_scores = []
        for word_loss in result:
            # start of a sentence
            token = queue_tokens.get()
            if token == "[CLS]":
                sentence_loss = 0.0
                word_count_per_sent = 0
            elif token == "[SEP]":
                new_line = 'uttid:,' + \
                            'preds:{},'.format(' '.join(list_tokens)) + \
                            'score_lm:{},'.format(' '.join(list_scores)) + \
                            'ppl:{:.2f}'.format(float(np.exp(sentence_loss / word_count_per_sent)))
                fw.write(new_line+'\n')
                list_tokens = []
                list_scores = []
            else:
                # add token
                list_tokens.append(tokenization.printable_text(token))
                list_scores.append('{:.3f}'.format(np.exp(-word_loss[0])))

                sentence_loss += word_loss[0]
                word_count_per_sent += 1


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # tf.gfile.MakeDirs(FLAGS.output_dir)

    run_config = tf.contrib.tpu.RunConfig(
            cluster=None, master=None,
            # model_dir=FLAGS.output_dir,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    num_shards=8,
                    per_host_input_for_training=3))

    model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint)

    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.predict_batch_size)

    # predict_examples = read_examples(FLAGS.input_file)
    # features, all_tokens = convert_examples_to_features(predict_examples, FLAGS.max_seq_length, tokenizer)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

    dataset = TextDataSet(FLAGS.input_file, FLAGS.vocab_file, FLAGS.max_seq_length)

    # result = estimator.predict(input_fn=predict_input_fn)
    def predict_input_fn(params):
        batch_size = params['batch_size']

        d = tf.data.Dataset.from_generator(
            lambda: dataset,
            {
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                "segment_ids": tf.int32,
                "masked_lm_positions": tf.int32,
                "masked_lm_ids": tf.int32
            },
            {
                "input_ids": tf.TensorShape([None]),
                "input_mask": tf.TensorShape([None]),
                "segment_ids": tf.TensorShape([None]),
                "masked_lm_positions": tf.TensorShape([None]),
                "masked_lm_ids": tf.TensorShape([None])
            }).batch(8)
        
        return d

    result = estimator.predict(input_fn=predict_input_fn)
    score(result, dataset.queue_tokens, FLAGS.output)


if __name__ == "__main__":
    tf.app.run()
