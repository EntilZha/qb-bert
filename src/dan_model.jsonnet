function(lr=0.001, dropout=0.5, debug=false) {
  dataset_reader: {
    qanta_path: '/fs/clip-quiz/entilzha/code/qb-bert/src/data/qanta.mapped.2018.04.18.json',
    lazy: false,
    debug: debug,
    type: 'qanta',
    full_question_only: false,
    first_sentence_only: false,
    char_skip: null,
    tokenizer: {
      type: 'word',
    },
    token_indexers: {
      text: {
        type: 'single_id',
        lowercase_tokens: true
      }
    },
  },
  train_data_path: 'guesstrain',
  validation_data_path: 'guessdev',
  model: {
    type: 'dan_guesser',
    dropout: dropout,
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['text', 'num_tokens']],
    batch_size: 512,
  },
  trainer: {
    type: 'callback',
    callbacks: [
      {
        type: 'checkpoint',
        checkpointer: { num_serialized_models_to_keep: 1 },
      },
      { type: 'track_metrics', patience: 3, validation_metric: '+accuracy' },
      'validate',
      { type: 'log_to_tensorboard' },
      {
        type: 'update_learning_rate',
        learning_rate_scheduler: {
          type: 'reduce_on_plateau',
          patience: 2,
          mode: 'max',
          verbose: true,
        },
      },
    ] + if debug then [] else [{ type: 'log_to_comet', project_name: 'qb-bert' }],
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 50,
    cuda_device: 0,
  },
}
