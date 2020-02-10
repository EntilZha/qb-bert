function(lr=0.00001) {
  dataset_reader: {
    lazy: false,
    debug: false,
    type: 'qanta',
    fold: 'guesstrain',
    break_questions: true,
  },
  train_data_path: 'data/qanta.guesstrain-90.2018.04.18.json',
  validation_data_path: 'data/qanta.guesstrain-10.2018.04.18.json',
  model: {
    type: 'guesser',
    dropout: 0.5,
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['text', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    validation_metric: '+accuracy',
    num_serialized_models_to_keep: 1,
    num_epochs: 50,
    patience: 2,
    cuda_device: 0,
  },
}
