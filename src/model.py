import torch.nn as nn

def get_activation_fn(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class NewsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn_name="relu"):
        super(NewsClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_fn(activation_fn_name),
            nn.Dropout(0.5), # Regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation_fn(activation_fn_name),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # We don't apply softmax here because CrossEntropyLoss applies it internally
        return self.network(x)