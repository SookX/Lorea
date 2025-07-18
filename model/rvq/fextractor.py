import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 

class EPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.min = nn.Parameter(torch.randint(low=-5, high=-1, size=(1,), dtype=torch.float32))
        self.max = nn.Parameter(torch.randint(low=1, high=5, size=(1,), dtype=torch.float32))

        self.k = nn.Parameter(torch.rand(1, dtype=torch.float32))
    
    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        return torch.exp(self.k * x)  
    
class SincBlock(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.sinc_conv = SincConv_fast(c, k)
        self.batch_norm = nn.BatchNorm1d(c)
        self.pool = nn.MaxPool1d(2, 2)
    def log_compression(self, x):
        return torch.log(torch.abs(x) + 1)
    
    def forward(self, x):
        x = self.sinc_conv(x)
        

        x = self.log_compression(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        return x
    
class DConvBlock(nn.Module):
    def __init__(self, i, c, k, s, dropout = True):
        super().__init__()
        self.depthwise = nn.Conv1d(i, i, kernel_size=k, padding=k//2, stride=s, groups=i)
        self.pointwise = nn.Conv1d(i, c, kernel_size=1)
        self.activation = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(c)
        self.is_dropout = dropout
        self.dropout_layer = nn.Dropout1d(0.15)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        if self.is_dropout:
            x = self.dropout_layer(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc_block = SincBlock(128, 52)

        self.d_conv_1 = DConvBlock(128, 128, 25, 2, False)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout1d(0.1)
        self.d_conv_2 = DConvBlock(128, 256, 9, 1, False)
        self.d_conv_3 = DConvBlock(256, 256, 9, 1, False)
        
        #self.d_conv_4 = DConvBlock(256, 256, 9, 1, False)
        #self.gpool = nn.AdaptiveAvgPool1d(1)
        #self.ff = nn.Linear(256, 35)
    def forward(self, x):
        skip_connections = []
        skip_connections.append(x)
        x = self.sinc_block(x)
        skip_connections.append(x)
        x = self.d_conv_1(x)
        x = self.pool(x)
        x = self.dropout(x)
        skip_connections.append(x)
        
        x = self.d_conv_2(x)
        x = self.pool(x)
        skip_connections.append(x)
        
        x = self.d_conv_3(x)
        x = self.pool(x)
        skip_connections.append(x)
        #x = self.d_conv_4(x)
        #x = self.pool(x)
        #x = self.gpool(x).squeeze(-1)
        #x = self.ff(x)
        return x, skip_connections
        
#class FeatureExtractor(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.sinc_module = SincConv_fast(16, 3, in_channels=in_channels)
#        self.pooling = nn.MaxPool1d(2, 2)
#        self.batch_norm_1 = nn.BatchNorm1d(16)
#        self.batch_norm_2 = nn.BatchNorm1d(32)
#        self.batch_norm_3 = nn.BatchNorm1d(64)
#        self.batch_norm_4 = nn.BatchNorm1d(out_channels)
#        self.activation_1 = nn.LeakyReLU()
#        self.activation_2 = nn.LeakyReLU()
#        self.activation_3 = nn.LeakyReLU()
#        self.dropout = nn.Dropout1d(0.2)
#        self.conv_layer_1 = nn.Conv1d(16, 32, 5, 1)
#        self.conv_layer_2 = nn.Conv1d(32, 64, 5, 1)
#        self.conv_layer_3 = nn.Conv1d(64, 128, 5, 1)
#        self.conv_layer_4 = nn.Conv1d(128, out_channels, 5, 1)
#        self.gpool = nn.AdaptiveAvgPool1d(1)
#        self.ff = nn.Linear(out_channels, 35)
#
#
#    def forward(self, x):
#        x = self.sinc_module(x)
#        x = self.pooling(x)
#        x = self.batch_norm_1(x)
#        x = self.activation_1(x)
#        x = self.dropout(x)
#        x = self.conv_layer_1(x)
#        x = self.pooling(x)
#        x = self.batch_norm_2(x)
#        x = self.activation_2(x)
#        x = self.dropout(x)
#        x = self.conv_layer_2(x)
#        x = self.pooling(x)
#        x = self.batch_norm_3(x)
#        x = self.activation_3(x)
#        x = self.dropout(x)
#        x = self.conv_layer_3(x)
#        x = self.pooling(x)
#        x = self.batch_norm_4(x)
#        x = self.activation_3(x)
#        x = self.gpool(x).squeeze(-1)
#        x = self.ff(x)
#        return x
class TDConvBlock(nn.Module):
    def __init__(self, i, c, k, s, dropout = True):
        super().__init__()
        self.pointwise_reverse = nn.ConvTranspose1d(i, i, kernel_size=1, stride=1)
        self.depthwise_reverse = nn.ConvTranspose1d(i, c, kernel_size=k, 
                                                    padding=k//2, stride=s, groups=c)
        self.batch_norm = nn.BatchNorm1d(c)
        self.activation = nn.LeakyReLU()
        self.is_dropout = dropout
        self.dropout_layer = nn.Dropout1d(0.15)
    
    def forward(self, x):
        x = self.pointwise_reverse(x)
        x = self.depthwise_reverse(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        if self.is_dropout:
            x = self.dropout_layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.td_conv_1 = TDConvBlock(256, 256, 9, 1)
        self.td_conv_2 = TDConvBlock(256, 128, 9, 1)
        self.td_conv_3 = TDConvBlock(128, 128, 25, 2, False)
        self.dropout = nn.Dropout1d(0.1)
        self.final_proj = nn.Conv1d(128, 1, kernel_size=52, stride=1, dilation=1, padding=0)
        
    def skip_connection(self, x, skip_connection):

        diff = skip_connection.shape[-1] - x.shape[-1]

        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip_connection.shape[-1]]

        return x + skip_connection

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        x = self.skip_connection(x, skip_connections[0])
        x = self.upsample(x)
        x = self.td_conv_1(x)
        x = self.skip_connection(x, skip_connections[1])
        x = self.upsample(x)
        x = self.td_conv_2(x)
        x = self.skip_connection(x, skip_connections[2])
        x = self.dropout(x)
        x = self.upsample(x)
        x = self.td_conv_3(x)
        x = self.skip_connection(x, skip_connections[3])
        x = self.upsample(x)
        x = self.final_proj(x)
        x = self.skip_connection(x, skip_connections[4])
        return x
if __name__ == "__main__":
    x = torch.rand(4, 1, 160000)
    model = FeatureExtractor()
    y = model(x)

    decoder = Decoder()
    z = decoder(y)
    #print(z.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(y.shape)
    print(f"Total parameters: {pytorch_total_params}")