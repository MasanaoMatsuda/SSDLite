from torch import nn
import math

def build_conv_bn(input_c, output_c, stride, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_c),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 3, stride, 1, bias=False),
            nn.ReLU6(inplace=True)
        )


def build_conv_1x1_bn(input_c, output_c, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_c),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True)
        )
    

def build_bottleneck(input_c, output_c, stride, expand_ratio, use_batch_norm=True):
    assert stride in [1, 2]
    
    if expand_ratio == 1:
        return SpatialSeparableConv2d(input_c, output_c, stride, use_batch_norm)
    else:
        return InvertedResidual(input_c, output_c, stride, expand_ratio, use_batch_norm)


class SpatialSeparableConv2d(nn.Module):
    def __init__(self, input_c, output_c, stride, use_batch_norm=True):
        super().__init__()
        
        self.use_res_connect = (stride == 1 and input_c == output_c)
        
        if use_batch_norm:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(input_c, input_c, 3, stride, 1, groups=input_c, bias=False),
                nn.BatchNorm2d(input_c),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_c),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(input_c, input_c, 3, stride, 1, groups=input_c, bias=False),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        
    
class InvertedResidual(nn.Module):
    def __init__(self, input_c, output_c, stride, expand_ratio, use_batch_norm=True):
        super().__init__()
        
        self.use_res_connect = (stride == 1 and input_c == output_c)

        hidden_c = round(input_c * expand_ratio)
        
        if use_batch_norm:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(input_c, hidden_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_c, hidden_c, 3, stride, 1, groups=hidden_c, bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv2d(hidden_c, output_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_c),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(input_c, hidden_c, 1, 1, 0, bias=False),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_c, hidden_c, 3, stride, 1, groups=hidden_c, bias=False),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv2d(hidden_c, output_c, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2, use_batch_norm=True):
        super().__init__()
        assert input_size % 32 == 0
          
        INPUT_C = 32
        FINAL_C = 1280
        INVERTED_RESIDUAL_SETTING = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_c = int(INPUT_C * width_mult)
        final_c = int(FINAL_C * width_mult) if width_mult > 1.0 else FINAL_C
        
        # Building Feature Extractors
        self.feature_extractor = []
        
        # First Block Layers
        self.feature_extractor.append(build_conv_bn(input_c=3, output_c=input_c, stride=2))
        
        # Inverted Residual Blocks
        for t, c, n, s in INVERTED_RESIDUAL_SETTING:
            output_c = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.feature_extractor.append(build_bottleneck(input_c, output_c, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm))
                else:
                    self.feature_extractor.append(build_bottleneck(input_c, output_c, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm))
                input_c = output_c
                
        # Final Block Layers
        self.feature_extractor.append(build_conv_1x1_bn(input_c, final_c, use_batch_norm=use_batch_norm))
        
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self._initialize_weights()
        self.out_channels = [576, 1280, 512, 256, 256, 64]
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()