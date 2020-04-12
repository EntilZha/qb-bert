conda create -n lm
conda install python=3.7
conda install -c pytorch pytorch
pip install transformers tensorboardX click
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm

# Run Command
# export TRAIN_FILE=/fs/clip-quiz/entilzha/code/qb-bert/src/data/qanta.train.raw
# export TEST_FILE=/fs/clip-quiz/entilzha/code/qb-bert/src/data/qanta.dev.raw
# python run_language_modeling.py --output_dir /tmp/qanta_lm --model_type=bert --model_name_or_path=bert-base-uncased --do_train --train_data_file=$TRAIN_FILE --do_eval --eval_data_file=$TEST_FILE --mlm