import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        return self.conv(x)


class AudioSEDNet(nn.Module):
    """
    Lightweight 2D CNN for Sound Event Detection (SED) & Feature Extraction.
    Accepts log-Mel Spectrogram input of shape (B, 1, 64, T).
    """
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Linear(256, embedding_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_embeddings(self, x):
        """Extract bottleneck feature embedding for clustering / anomaly detection."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        embeddings = F.relu(self.fc_embedding(x))
        return embeddings

    def forward(self, x):
        embeddings = self.extract_embeddings(x)
        logits = self.classifier(self.dropout(embeddings))
        return logits
