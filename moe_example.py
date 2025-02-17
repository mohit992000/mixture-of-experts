import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Each expert is a simple Linear layer in this example
        self.experts = nn.ModuleList([
            nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)
        ])

        # The router produces logits indicating how relevant each expert is
        self.router = nn.Linear(expert_dim, num_experts)

    def forward(self, x):
        """
        Args:
          x: Tensor of shape [batch_size, expert_dim]
        Returns:
          out: Tensor of shape [batch_size, expert_dim],
               the mixture of top-k experts' outputs.
        """
        batch_size, expert_dim = x.shape

        # 1) Compute softmax over experts
        routing_logits = self.router(x)                  # [batch_size, num_experts]
        weights = F.softmax(routing_logits, dim=-1)      # [batch_size, num_experts]

        # 2) Select top-k experts for each sample
        #    top_k_values, top_k_indices: shape [batch_size, top_k]
        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=-1)

        # 3) Prepare an output buffer
        out = torch.zeros_like(x)  # will accumulate outputs for each sample

        # 4) Group by expert index
        #
        # For each expert 'e', find which (sample, k) pairs picked that expert.
        # Then process all these samples in one batched forward pass.
        for e in range(self.num_experts):
            # mask: True/False where top_k_indices == e
            # shape: [batch_size, top_k]
            mask = (top_k_indices == e)

            # positions: list of [batch_idx, k_idx]
            positions = mask.nonzero(as_tuple=False)

            if positions.size(0) == 0:
                continue

            chosen_batch_indices = positions[:, 0]
            chosen_k_indices     = positions[:, 1]

            # Gather the inputs for these samples
            chosen_inputs = x[chosen_batch_indices]  # shape [N, expert_dim]

            # Apply the expert to the gathered inputs
            chosen_outputs = self.experts[e](chosen_inputs)  # shape [N, expert_dim]

            # Gather the routing weights for these samples
            chosen_weights = top_k_values[chosen_batch_indices, chosen_k_indices]  # shape [N]

            # Scale the expert output by its weight
            chosen_outputs *= chosen_weights.unsqueeze(-1)

            # Scatter-add the results back into 'out'
            out.index_add_(0, chosen_batch_indices, chosen_outputs)

        return out


def main():
    # Example usage
    # Let's create a batch of size 4, each input of dimension 8
    batch_size = 4
    expert_dim = 8
    num_experts = 3
    top_k = 2

    model = MoELayer(num_experts=num_experts, expert_dim=expert_dim, top_k=top_k)

    # Create a random input
    x = torch.randn(batch_size, expert_dim)

    print("Input x:")
    print(x)

    # Forward pass
    output = model(x)

    print("\nOutput of MoELayer:")
    print(output)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import your existing MoELayer from your 'moe_example.py' (adjust the path as needed)
# from moe_example import MoELayer

###############################################
# If you haven't split your code, here's a condensed MoELayer:
###############################################
class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Each expert is a simple Linear layer
        self.experts = nn.ModuleList([
            nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)
        ])

        # Router to produce logits over experts
        self.router = nn.Linear(expert_dim, num_experts)

    def forward(self, x):
        """
        x: [batch_size, expert_dim]
        Returns: [batch_size, expert_dim]
        """
        batch_size, expert_dim = x.shape

        # 1) Compute expert weights
        routing_logits = self.router(x)                  # [batch_size, num_experts]
        weights = F.softmax(routing_logits, dim=-1)      # [batch_size, num_experts]

        # 2) Pick top-k experts per sample
        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=-1)

        # 3) Prepare output buffer
        out = torch.zeros_like(x)

        # 4) Group by expert index
        for e in range(self.num_experts):
            mask = (top_k_indices == e)
            positions = mask.nonzero(as_tuple=False)

            if positions.size(0) == 0:
                continue

            chosen_batch_indices = positions[:, 0]
            chosen_k_indices = positions[:, 1]

            chosen_inputs = x[chosen_batch_indices]
            chosen_outputs = self.experts[e](chosen_inputs)

            chosen_weights = top_k_values[chosen_batch_indices, chosen_k_indices]
            chosen_outputs *= chosen_weights.unsqueeze(-1)

            out.index_add_(0, chosen_batch_indices, chosen_outputs)

        return out
###############################################

class MoEClassifier(nn.Module):
    def __init__(self, num_experts, expert_dim, top_k, num_classes=10):
        super().__init__()
        # Our MoE layer
        self.moe = MoELayer(num_experts, expert_dim, top_k)
        # Final linear classification
        self.classifier = nn.Linear(expert_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, expert_dim]
        moe_out = self.moe(x)            # [batch_size, expert_dim]
        logits = self.classifier(moe_out)  # [batch_size, num_classes]
        return logits


def train_moe_mnist():
    # Hyperparameters
    batch_size = 64
    expert_dim = 784   # 28x28 images, flattened
    num_experts = 3
    top_k = 2
    num_classes = 10   # 10 digits in MNIST
    lr = 0.001
    num_epochs = 5

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load MNIST
    # Transform: convert images to tensors and flatten 28x28 -> 784
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 2) Initialize MoE-based classifier
    model = MoEClassifier(num_experts, expert_dim, top_k, num_classes).to(device)

    # 3) Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # (Optional) Check accuracy on training set or a validation set

    # 5) Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train_moe_mnist()