# This iis the networks script
import torch
import torch.nn as nn


import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from networks import *
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

import torch
import torch.nn as nn


class MultiHeadAttentionPrompt(nn.Module):
    def __init__(self, in_dim, use_softmax, use_tanh, num_heads=5):
        super(MultiHeadAttentionPrompt, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        # Query, Key, Value Linear layers
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

        # Output Linear layer
        self.out = nn.Linear(in_dim, in_dim)
        self.use_softmax = use_softmax
        self.use_tanh = use_tanh

    def forward(self, target, source_prompts):
        batch_size = target.size(0)  # 60
        prompt_num = source_prompts.size(0)  # 31

        # Reshape target and source prompts for attention
        Q = self.query(target.permute(0, 2, 3, 1)).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.key(source_prompts.permute(0, 2, 3, 1)).view(prompt_num, -1, self.num_heads, self.head_dim)
        V = self.value(source_prompts.permute(0, 2, 3, 1)).view(prompt_num, -1, self.num_heads, self.head_dim)

        # Attention scores
        attention_scores = torch.einsum("bqhd,pqhd->bpqh", Q, K) / (self.head_dim ** 0.5)
        if self.use_softmax:
            attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        if self.use_tanh:
            attention_scores = torch.tanh(attention_scores)
        attention_weights = attention_scores
        # Weighted sum of values
        Z = torch.einsum("bpqh,pqhd->bqhd", attention_weights, V)

        # Combine heads and output
        Z_f = Z.view(batch_size, -1, self.num_heads * self.head_dim)
        Z_f = self.out(Z_f)

        # Reshape Z_f to match target dimensions and add to target
        Z_f = Z_f.view(batch_size, 9, 9, self.head_dim * self.num_heads).permute(0, 3, 1, 2)
        T_prime = target + Z_f
        return T_prime


class VAE(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=128, z_dim=256):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # Size remains [9, 9]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Size remains [9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Size remains [9, 9]
            nn.ReLU()
        )
        # Flatten and apply Linear layers for VAE latent space
        self.fc1 = nn.Linear(128 * 9 * 9, z_dim)  # Adjusted for 9x9 feature maps
        self.fc2 = nn.Linear(128 * 9 * 9, z_dim)
        self.fc3 = nn.Linear(z_dim, 128 * 9 * 9)
        self.inputs_head = nn.Linear(z_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # Size remains [9, 9]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # Size remains [9, 9]
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=1, padding=1),  # Output size [9, 9]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z)).view(z.size(0), 128, 9, 9)
        return self.decoder(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = self.inputs_head(mu)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, is_nonlinear=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, 256)  # Assuming input_height and input_width are defined
        self.readout = nn.Linear(256, out_features)
        #self.DistributionUncertainty = DistributionUncertainty()

    def forward(self, x):
        # return torch.nn.Linear(in_features, out_features)
        #x = self.DistributionUncertainty(x)
        x = F.dropout(F.selu(self.fc(x)), p=0.5)
        x = self.readout(x)
        return x


class PGDG(nn.Module):
    def __init__(self, domain_num, num_classes=4):
        super(PGDG, self).__init__()

        self.featurizer = VAE()
        self.com = False
        self.gamma = 2
        self.lf_alg = 'entropy'
        self.test_conf = 0.5
        self.num_domains = domain_num
        feature_num = 256
        # self.fc = nn.Linear(feature_num, 1024)  # Assuming input_height and input_width are defined
        #self.readout = nn.Linear(1024, num_classes)
        self.classifier_list = nn.ModuleList([
            Classifier(
                feature_num,
                num_classes,
                is_nonlinear=False)
            for i in range(self.num_domains + 1)
        ])
        self.num_class = num_classes

        self.network = nn.Sequential(self.featurizer, self.classifier_list)

    def softmax_entropy(self, x: torch.Tensor):
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def forward(self, x):

        rec_x, mu, logvar = self.featurizer(x)
        entropy = torch.tensor(1e10)
        result = None
        ents, y_hats = [], []
        for i in range(self.num_domains + 1):
            y_hat = self.classifier_list[i](mu)
            ent = self.softmax_entropy(y_hat).mean()
            ents.append(ent.item())
            if ent < entropy:
                entropy = ent
                result = y_hat

        return result
