# A script that runs a tokenizer on a text file with one sentence per line.
#
# Example usage:
#  bazel build syntaxnet:parser_eval
#  cat untokenized-sentences.txt |
#    syntaxnet/models/parsey_universal/tokenize.sh \
#    $MODEL_DIRECTORY > output.conll
#
# Models can be downloaded from
#  http://download.tensorflow.org/models/parsey_universal/<language>.zip
# for the languages listed at
#  https://github.com/tensorflow/models/blob/master/syntaxnet/universal.md
#

PARSER_TRAIN=bazel-bin/syntaxnet/parser_trainer
PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
CONTEXT=syntaxnet/models/parsey_universal/context.pbtxt
INPUT_FORMAT=stdin-untoken
MODEL_DIR=$1
PLANG=$2

PARAMS=200x200-0.08-4400-0.85-4

python3 bazel-bin/syntaxnet/parser_trainer \
    --arg_prefix=brain_translator \
    --batch_size=32 \
    --decay_steps=4400 \
    --graph_builder=greedy \
    --hidden_layer_sizes=200,200 \
    --learning_rate=0.08 \
    --momentum=0.85 \
    --output_path=$MODEL_DIR \
    --task_context=$CONTEXT \
    --seed=4 \
    --training_corpus=nc-mrg-fr-en-train \
    --tuning_corpus=nc-mrg-fr-en-dev \
    --params=$PARAMS \
    --alsologtostderr \
    --report_every=100 \
    --checkpoint_every=100 \
    --num_epochs=10 
