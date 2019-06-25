# File for running training on Oct dataset
# TRAIN and VAL must be tab separated files containing sequence strings and intensity values
# TRAIN and VAL are produced by splitting array1 in the ratio 70:30
TRAINDATA=
VALDATA=

python motif_find.py  \
    --long_term_recurrence \
    --train_data $TRAINDATA \
    --test_data $VALDATA \
    --output_prefix Oct1 \
    --num_epochs 150  \
    --cuda \
    --batch_size 100  \
    --preactivation quadruple \
    --num_long_term_states 128  \
    --use_likelihood_ratio  \
    --num_motifs 5  \
    --motif_length 11   \
    --learning_rate 0.001 \
    --use_log_forward
