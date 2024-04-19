class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config.pretrained:
            self.model = AutoModel.from_pretrained(self.config.text_encoder_model)
        else:
            self.model = AutoModel.from_config(self.config.text_encoder_model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_tokenizer)

        for p in self.model.parameters():
            p.requires_grad = False  # Set requires_grad to False for all parameters

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, texts):
        input_ids, attention_mask = self.tokenize_texts(texts)
        embeddinbgs = self.inputs_to_embeddings(input_ids, attention_mask)
        return embeddinbgs

    def tokenize_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].detach().to(self.config.device)
        attention_mask = inputs['attention_mask'].detach().to(self.config.device)
        return input_ids, attention_mask

    def inputs_to_embeddings(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :].detach()