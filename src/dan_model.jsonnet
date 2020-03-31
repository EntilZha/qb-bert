function(n_wiki_sentences=0, lr=0.001, dropout=0.5, debug=false, hidden_dim=300, pool="avg", n_hidden_layers=1, pytorch_seed=0, numpy_seed=0, random_seed=0) {
  pytorch_seed: pytorch_seed,
  numpy_seed: numpy_seed,
  random_seed: random_seed,
  dataset_reader: {
    qanta_path: '/fs/clip-quiz/entilzha/code/qb-bert/src/data/qanta.mapped.2018.04.18.json',
    wiki_path: '/fs/clip-quiz/entilzha/code/qb-bert/src/data/wiki_lookup.json',
    lazy: false,
    debug: debug,
    type: 'qanta',
    full_question_only: false,
    first_sentence_only: false,
    char_skip: null,
    n_wiki_sentences: n_wiki_sentences,
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
    hidden_dim: hidden_dim,
    n_hidden_layers: n_hidden_layers,
    pool: pool,
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
