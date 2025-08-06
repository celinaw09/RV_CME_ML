import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    total_loss = 0
    correct = 0
    num_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # assuming raw logits
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        num_samples += data.size(0)

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    avg_loss = total_loss / num_samples
    accuracy = 100. * correct / num_samples
    print(f"Train Epoch {epoch} Average loss: {avg_loss:.4f}, Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)")
    return avg_loss, accuracy