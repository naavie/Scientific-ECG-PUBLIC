def train_epoch(model, loader, optimizer, classes):
    tqdm_object = tqdm(loader, total=len(loader))
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    for batch in tqdm_object:
        model.train()
        batch = {k: v.to(CONFIG.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss, image_embeddings, text_embeddings = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            class_embeddings = model.text_to_embeddings(classes)

        accuracy = calc_accuracy(image_embeddings, batch['caption'], class_embeddings, classes)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_accuracy=accuracy_meter.avg)

    return loss_meter, accuracy_meter