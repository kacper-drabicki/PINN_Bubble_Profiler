import torch

def train(model, loss_fn, optimizer, epochs, scheduler, config):
    device = torch.device(config.device)
    model.train()
    
    r  = torch.linspace(0.01, config.r_max, 500, device=device).view(-1,1).requires_grad_(True)

    optimizer = optimizer(model.parameters())
    scheduler = scheduler(optimizer)
    for epoch in range(epochs):
        optimizer.zero_grad()

        y = model(r)

        loss = loss_fn(r, y)

        loss.backward()
        
        optimizer.step()
        scheduler.step(loss)

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}; Loss: {loss.item():.10e}')

def pretrain(model, config):
    return train(
        model,
        loss_fn=config.pretrain_loss_fn,
        optimizer=config.pretrain_optimizer,
        epochs=config.pretrain_epochs,
        scheduler=config.pretrain_scheduler,
        config=config
    )

def finetune(model, config):
    return train(
        model,
        loss_fn=config.finetune_loss_fn,
        optimizer=config.finetune_optimizer,
        epochs=config.finetune_epochs,
        scheduler=config.finetune_scheduler,
        config=config
    )
