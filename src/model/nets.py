import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer


class SpStGCN(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(SpStGCN, self).__init__()

        self.num_input, num_channel, _, _, _ = data_shape
        self.block_args = block_args

        # input branches
        self.input_branches = SpStGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs)

        # main stream
        # last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        # self.main_stream = EfficientGCN_Blocks(
        #     init_channel = self.num_input * last_channel,
        #     block_args = block_args[fusion_stage:],
        #     **kwargs
        # )

        # output
        last_channel = self.num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = SpStGCN_Classifier(last_channel, **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x, Ad):
        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)

        # input branches
        x_mid= []
        for i in range(self.num_input):
            x_branch = self.input_branches(x[i], Ad)     # 需要3个branch fuse的时候，用Ad[i]
            x_mid.append(x_branch)
        x = torch.cat(x_mid, dim=1)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)

        return out, feature


class SpStGCN_Blocks(nn.Module):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(SpStGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size
        self.block_args = block_args
        self.input_channel = input_channel

        self.bn = nn.BatchNorm2d(input_channel)
        self.spatial_graph_layer = Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs)
        self.Temporal_Basic_Layer = Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs)

        last_channel = init_channel
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')

        channel0, stride0, depth0 = block_args[0]
        self.spatial_graph_layer0 = Spatial_Graph_Layer(last_channel, channel0, max_graph_distance, **kwargs)
        self.temporal_layer0 = temporal_layer(channel0, temporal_window_size, stride=stride0 if depth0 == 0 else 1, **kwargs)
        self.att0 = Attention_Layer(channel0, **kwargs)

        channel1, stride1, depth1 = block_args[1]
        self.spatial_graph_layer1 = Spatial_Graph_Layer(channel0, channel1, max_graph_distance, **kwargs)
        self.temporal_layer1 = temporal_layer(channel1, temporal_window_size, stride=stride1 if depth1 == 0 else 1, **kwargs)
        self.att1 = Attention_Layer(channel1, **kwargs)

        channel2, stride2, depth2 = block_args[2]
        self.spatial_graph_layer2 = Spatial_Graph_Layer(channel1, channel2, max_graph_distance, **kwargs)
        self.temporal_layer2 = temporal_layer(channel2, temporal_window_size, stride=stride2 if depth2 == 0 else 1, **kwargs)
        self.att2 = Attention_Layer(channel2, **kwargs)

        channel3, stride3, depth3 = block_args[3]
        self.spatial_graph_layer3 = Spatial_Graph_Layer(channel2, channel3, max_graph_distance, **kwargs)
        self.temporal_layer3 = temporal_layer(channel3, temporal_window_size, stride=stride3 if depth3 == 0 else 1, **kwargs)
        self.att3 = Attention_Layer(channel3, **kwargs)

    def forward(self, x, Ad):
        if self.input_channel > 0:  # input branch
            # print("input branch", x.size(), Ad.size())
            x1 = self.bn(x)
            x2 = self.spatial_graph_layer(x1, Ad)
            x3 = self.Temporal_Basic_Layer(x2)

            x4 = self.spatial_graph_layer0(x3, Ad)
            x5 = self.temporal_layer0(x4)
            x6 = self.att0(x5)

            x7 = self.spatial_graph_layer1(x6, Ad)
            x8 = self.temporal_layer1(x7)
            x9 = self.att1(x8)

            x10 = self.spatial_graph_layer2(x9, Ad)
            x11 = self.temporal_layer2(x10)
            x12 = self.att2(x11)

            x13 = self.spatial_graph_layer3(x12, Ad)
            x14 = self.temporal_layer3(x13)
            x15 = self.att3(x14)

        else:  # main stream
            # print("main stream", x.size(), Ad.size())
            x4 = self.spatial_graph_layer0(x, Ad)
            x5 = self.temporal_layer0(x4)
            x6 = self.att0(x5)

            x7 = self.spatial_graph_layer1(x6, Ad)
            x8 = self.temporal_layer1(x7)
            x15 = self.att1(x8)

        return x15




class SpStGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(SpStGCN_Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
