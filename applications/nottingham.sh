# Example file showing music modeling training for nottingham dataset

# Set correct paths
VALDATA=
TRAINDATA=

python ml_train_music.py \
    --val_data $VALDATA \
    --train_data $TRAINDATA \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --batch_size 8 \
    --dimension 88 \
    --dropout 0.5 \
    --num_stochastic_nodes 16 \
    --num_recurrent_layers 1 \
    --num_recurrent_units 512 \
    --num_dense_layers 1 \
    --num_dense_units 512 \
    --clip_gradients \
    --entropy_regularizer \
    --heating_iterations 10 \
    --cooling_iterations 0 \
    --activation relu \
    --cuda \
    --output_prefix nottingham \
    --order 1 \
    --early_stop
