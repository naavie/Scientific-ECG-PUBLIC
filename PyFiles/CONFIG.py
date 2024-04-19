class CONFIG:
    debug = False
    batch_size = 128
    num_workers = 2
    head_lr = 0.001
    image_encoder_lr = 0.001
    patience = 5
    factor = 0.8
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Model
    # model_name = 'resnet18'
    image_embedding_size = 512

    # Text Model
    text_encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
    text_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'
    text_embedding_size = 768
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 10.0
    optimizer = torch.optim.Adam

    # image size
    size = 224

    # for projection head; used for both image and text encoder
    num_projection_layers = 1
    projection_dim = 64  # Adjust as needed
    dropout = 0.0
    ecg_sr = 128