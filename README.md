
# Output Folder Structure - Complete trainning on Kaggle (approximately 9 hours)

```directory
/kaggle/working/
├── 📁 models/                        # Thư mục lưu trữ trọng số mô hình (weights)
│   ├── best_a1.pth                   # Checkpoint tốt nhất của Model A1 (LSTM decoder)
│   ├── best_a2.pth                   # Checkpoint tốt nhất của Model A2 (Transformer decoder)
│   ├── best_b2.pth                   # Checkpoint tốt nhất của Model B2 (BLIP fine-tuned)
│   │                                 # (Chỉ khởi tạo khi RUN_BLIP_FINETUNE = True)
│   └── 📁 blip_processor/            # Bộ xử lý BLIP đã lưu (tokenizer, config...)
│       ├── preprocessor_config.json  # (Chỉ khởi tạo khi RUN_BLIP_FINETUNE = True)
│       ├── tokenizer_config.json
│       └── ...
├── 📁 outputs/                       # Thư mục chứa kết quả đầu ra
│   └── b1_predictions.csv            # Kết quả dự đoán của mô hình B1 (cột: pred, ref)
├── q_vocab.pkl                       # Từ điển câu hỏi (Vocab object, serialized)
├── a_vocab.pkl                       # Từ điển câu trả lời (Vocab object, serialized)
├── data_train_en.csv                 # Dataset huấn luyện đã dịch (VI → EN)
├── data_val_en.csv                   # Dataset kiểm định đã dịch (VI → EN)
└── data_test_en.csv                  # Dataset kiểm thử đã dịch (VI → EN)
                                      # (Các file .csv chỉ tạo khi RUN_PRETRANSLATE = True)
