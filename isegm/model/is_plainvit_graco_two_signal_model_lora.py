from isegm.utils.serialization import serialize
from .is_model import ISModel
from .is_plainvit_graco_model_lora import SimpleFPN
from .modeling.models_vit_dual_lora import VisionTransformer_duallora, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .modeling.clip_text_encoding import ClipTextEncoder


class PhraseCLIPGraCoModel_lora(ISModel):
    @serialize
    def __init__(
            self,
            backbone_params={},
            phrase_encoder_params={},
            neck_params={},
            head_params={},
            random_split=False,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer_duallora(**backbone_params)
        self.phrase_encoder = ClipTextEncoder(**phrase_encoder_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image, coord_features=None, gra=None, text=None):
        assert gra is None or text is None
        coord_features = self.patch_embed_coords(coord_features)
        B, _, _ = coord_features.shape

        text_features = None
        if text is not None:
            text_features = self.phrase_encoder(text).unsqueeze(1)  # [bs, 1, C]
            if text_features.size(0) != B:
                text_features = text_features.repeat(B, 1, 1)

        backbone_features = self.backbone.forward_backbone(image, coord_features, gra=gra, text_features=text_features,
                                                           shuffle=self.random_split)  # [bs, L, C]

        # Extract 4 stage image_encoder feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)
        return {'instances': self.head(multi_scale_features), 'instances_aux': None}
