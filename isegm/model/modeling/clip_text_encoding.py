import torch
from torch import nn
from .clip import clip

class ClipTextEncoder(nn.Module):
    def __init__(self, enocder_name="ViT-B/16", embedding_dim=512, out_dim=768):
        super().__init__()
        assert enocder_name in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                                'ViT-L/14@336px']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(enocder_name, device=self.device)

        # freeze model
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.phrase2gra_proj = nn.Linear(embedding_dim, out_dim)

    def forward(self, prompt):
        '''
        prompt: text tokens
        '''
        text_features = self.model.encode_text(prompt).type(torch.float32)
        # proj
        text_features = self.phrase2gra_proj(text_features)
        return text_features
