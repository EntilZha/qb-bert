function(lr=0.00001, dropout=0.25, pool='mean') {
  dataset_reader: {
    lazy: false,
    debug: true,
    type: 'qanta',
    fold: 'guesstrain',
    break_questions: true,
  },
  train_data_path: 'data/qanta.guesstrain-90.2018.04.18.json',
  validation_data_path: 'data/qanta.guesstrain-10.2018.04.18.json',
  model: {
    type: 'guesser',
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
      { type: 'log_to_comet', project_name: 'qb-bert' },
    ],
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 50,
    cuda_device: 0,
  },
}
