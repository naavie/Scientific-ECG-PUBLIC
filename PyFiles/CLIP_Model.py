class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CONFIG.projection_dim,
        dropout=CONFIG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CONFIG.temperature,
        image_embedding=CONFIG.image_embedding_size,
        text_embedding=CONFIG.text_embedding_size,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(CONFIG)
        self.text_encoder = TextEncoder(CONFIG)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_embeddings = self.image_to_embeddings(batch['image'])
        text_embeddings = self.text_to_embeddings(batch['caption'])

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean(), image_embeddings, text_embeddings

    def text_to_embeddings(self, texts):
        text_features = self.text_encoder(texts)
        text_embeddings = self.text_projection(text_features)
        return text_embeddings

    def image_to_embeddings(self, images):
        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features)
        return image_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def nxn_cos_sim(A, B, dim=1):
    a_norm = F.normalize(A, p=2, dim=dim)
    b_norm = F.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))