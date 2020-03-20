function(lr=0.00001, dropout=0.25, pool='mean', debug=false) {
  dataset_reader: {
    qanta_path: '/fs/clip-quiz/entilzha/code/qb-bert/src/data/qanta.mapped.2018.04.18.json',
    lazy: false,
    debug: debug,
    type: 'qanta',
    full_question_only: false,
    first_sentence_only: false,
    char_skip: null,
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: 'bert-base-uncased',
      do_lowercase: true,
      start_tokens: [],
      end_tokens: [],
    },
    token_indexers: {
      text: {
        type: 'bert-pretrained',
        pretrained_model: 'bert-base-uncased'
      }
    },
  },
  train_data_path: 'guesstrain',
  validation_data_path: 'guessdev',
  model: {
    type: 'bert_guesser',
    dropout: dropout,
    pool: pool,
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['text', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    type: 'callback',
    callbacks: [
      {
        type: 'checkpoint',
        checkpointer: { num_serialized_models_to_keep: 1 },
      },
      { type: 'track_metrics', patience: 2, validation_metric: '+accuracy' },
      'validate',
      { type: 'log_to_tensorboard' },
    ] + if debug then [] else [{ type: 'log_to_comet', project_name: 'qb-bert' }],
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 50,
    cuda_device: 0,
  },
}
