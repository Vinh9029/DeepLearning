Output after Trainning (approximately 9 hours)

/kaggle/working/
│
├── 📁 models/                        ← thư mục chứa model weights
│   ├── best_a1.pth                   ← best checkpoint Model A1 (LSTM decoder)
│   ├── best_a2.pth                   ← best checkpoint Model A2 (Transformer decoder)
│   ├── best_b2.pth                   ← best checkpoint Model B2 (BLIP fine-tuned)
│   │                                    * chỉ tạo nếu RUN_BLIP_FINETUNE = True
│   └── 📁 blip_processor/            ← BLIP processor đã save (tokenizer config...)
│       ├── preprocessor_config.json     * chỉ tạo nếu RUN_BLIP_FINETUNE = True
│       ├── tokenizer_config.json
│       └── ...
│
├── 📁 outputs/
│   └── b1_predictions.csv            ← predictions của B1 (cột: pred, ref)
│
├── q_vocab.pkl                       ← Question vocabulary (Vocab object, ~serialized)
├── a_vocab.pkl                       ← Answer vocabulary (Vocab object)
│
├── data_train_en.csv                 ← dataset train đã dịch VI→EN
├── data_val_en.csv                   ← dataset val đã dịch VI→EN      ┐ chỉ tạo nếu
└── data_test_en.csv                  ← dataset test đã dịch VI→EN     ┘ RUN_PRETRANSLATE = True
