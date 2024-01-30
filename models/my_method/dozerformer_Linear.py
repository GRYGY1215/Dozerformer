import torch
from torch import nn
from einops import rearrange

# Implementation from other source code.
from models.my_method.dozerformer_EncDec import dozerformer_Encoder, dozerformer_Decoder

from models.REVIN import RevIN
from models.my_method.build_model_util import series_decomp_multi

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mode = configs.mode
        assert self.mode in ["pretrain", 'finetune', "forecasting"], "Error mode."
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channel = configs.data_dim
        self.embed_dim = configs.embed_dim
        # self.mask_ratio = configs.mask_ratio
        self.encoder_depth = configs.encoder_depth
        self.decoder_depth = configs.decoder_depth
        self.decoder_embed_dim = configs.decoder_embed_dim
        self.dropout = configs.dropout
        self.epoch = 0
        configs.activation = 'gelu'

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)
        self.revin_layer_dec = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)

        # Seasonal encoder and decoder
        self.encoder_seasonal = dozerformer_Encoder(configs, mode='Seasonal')
        self.decoder_seasonal = dozerformer_Decoder(configs, mode='Seasonal')

        self.output_layer = nn.Conv2d(in_channels=self.embed_dim,
                                      out_channels=1,
                                      kernel_size=(1, 1))

        # Trend Encoder and Decoder
        self.trend_model = nn.Linear(configs.seq_len, configs.pred_len)


    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:
        x_enc = self.revin_layer(x_enc, 'norm')

        x_enc, trend_enc = self.decomp_multi(x_enc)
        x_dec, trend_dec = self.decomp_multi(x_dec)

        # Encoder
        encoder_output = self.encoder_seasonal(x_enc)
        decoder_output = self.decoder_seasonal(x_dec, encoder_output)

        # Inference from Decoder's output
        seasonal_predict = self.decoder_seasonal.decoder_segment.concat(decoder_output)
        seasonal_predict = self.output_layer(seasonal_predict)
        seasonal_predict = rearrange(seasonal_predict, 'b 1 seq_len ts_d -> b seq_len ts_d')[:, -self.pred_len:, :]


        # Trend
        trend_enc = rearrange(trend_enc, 'b seq_len ts_d -> b ts_d seq_len')
        trend_predict = self.trend_model(trend_enc)
        trend_predict = rearrange(trend_predict, 'b ts_d seq_len -> b seq_len ts_d')

        # Concate Trend and Seasonal
        final_predict = seasonal_predict + trend_predict

        # Inverse Revin
        final_predict = self.revin_layer(final_predict, 'denorm')
        return final_predict

