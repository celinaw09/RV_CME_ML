import torch
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            num_samples += data.size(0)

    avg_loss = test_loss / num_samples
    accuracy = 100. * correct / num_samples
    print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)\n")
    return avg_loss, accuracy