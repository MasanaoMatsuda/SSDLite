import torch
from torch import nn
from mobilenet_v2 import MobileNetV2, InvertedResidual


def build_SSeparableConv2d(input_c, output_c, kernel=1, stride=1, padding=0):
    """Replace Conv2d with a SpacialSeparableConvolution which is conposed depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(input_c, input_c, kernel, groups=input_c, stride=stride, padding=padding),
        nn.BatchNorm2d(input_c),
        nn.ReLU6(),
        nn.Conv2d(input_c, output_c, kernel_size=1),
    )


class Residual(InvertedResidual):
    def __init__(self, input_c, output_c, stride, expand_ratio, use_batch_norm=True):
        super().__init__(input_c, output_c, stride, expand_ratio, use_batch_norm)
    
    def forward(self, x):
        return super().forward(x)
    
    
class SSDLite(nn.Module):
    def __init__(self, num_label=21, width_mult=1.0):
        super().__init__()
        
        # Feature Extraction
        backbone = MobileNetV2()
        self.feature_extractor0 = nn.Sequential(*backbone.feature_extractor[:14], 
                                                *backbone.feature_extractor[14].conv[:3])
        self.feature_extractor1 = nn.Sequential(*backbone.feature_extractor[14].conv[3:],
                                                *backbone.feature_extractor[15:])
        
        self.additional_blocks = nn.ModuleList([
            Residual(1280, 512, stride=2, expand_ratio=0.2),
            Residual(512, 256, stride=2, expand_ratio=0.25),
            Residual(256, 256, stride=2, expand_ratio=0.5),
            Residual(256, 64, stride=2, expand_ratio=0.25)
        ])
        
        # Predict class label and box position
        self.num_label = num_label
        num_defaults = [6] * 6
        backbone.out_channels[0] = round(backbone.out_channels[0] * width_mult)
        self.loc = []
        self.conf = []
        for nd, oc in zip(num_defaults[:-1], backbone.out_channels[:-1]):
            self.loc.append(build_SSeparableConv2d(oc, nd*4, kernel=3, padding=1))
            self.conf.append(build_SSeparableConv2d(oc, nd*num_label, kernel=3, padding=1))
        self.loc.append(nn.Conv2d(backbone.out_channels[-1], num_defaults[-1]*4, kernel_size=1))
        self.conf.append(nn.Conv2d(backbone.out_channels[-1], num_defaults[-1]*num_label, kernel_size=1))
        self.conf = nn.ModuleList(self.conf)
        self.loc = nn.ModuleList(self.loc)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
    def bbox_view(self, src):
        ret = []
        for s, l, c in zip(src, self.loc, self.conf):
            loc = l(x).permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            conf = c(x).permute(0, 2, 3, 1).contiguous()
            conv = conf.view(conf.size(0), -1, self.num_label)
            ret.append((loc, conf))
        locs, confs = list(zip(*ret))
        
        return torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
    
    
    def forward(self, x):
        
        detection_feed = []
        
        x = self.feature_extractor0(x)
        detection_feed.append(x)
        
        x = self.feature_extractor1(x)
        detection_feed.append(x)
        
        x = self.feature_extractor(x)
        detection_feed = [x]
        for layer in self.additional_blocks:
            x = layer(x)
            detection_feed.append(x)
            
        return self.bbox_view(detection_feed)
    

    def load(self, state_dict):
        self.load_state_dict(
            torch.load(state_dict, map_location=lambda storage, loc: storage)
        )