import torch
import torch.nn as nn

class DT_Transformer(nn.Module):
    def __init__(
        self,
        device,
        emb_dim=1024,
        tf_layers=1,
        tf_head=2,
        tf_dim=128,
        activation="gelu",
        dropout=0.1,
        use_features=["images", "texts"],
        use_multiclass=False
    ):

        super().__init__()

        self.use_features = use_features
        self.emb_dim = emb_dim

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=tf_layers,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.gelu = nn.GELU()
        self.fcl = nn.Linear(self.emb_dim, self.emb_dim // 2)
        self.use_multiclass = use_multiclass

        if self.use_multiclass:
            self.output_score = nn.Linear(self.emb_dim // 2, 3)
        else:
            self.output_score = nn.Linear(self.emb_dim // 2, 1)

    def forward(self, img, txt):

        if "images" not in self.use_features:
            x = txt
        elif "texts" not in self.use_features:
            x = img
        else:
            x = torch.cat((img, txt), axis=1)

        b_size = x.shape[0]

        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fcl(x)
        x = self.gelu(x)
        x = self.dropout(x)
        y = self.output_score(x)
        return y