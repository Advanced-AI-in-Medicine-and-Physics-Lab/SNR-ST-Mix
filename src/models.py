from torch import nn
import timm


class GeneExpressionPredictor(nn.Module):
    def __init__(self, output_dim: int, backbone: str = "vit_large_patch16_224",
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, output_dim)

    def forward(self, images):
        return self.head(self.dropout(self.backbone(images)))

