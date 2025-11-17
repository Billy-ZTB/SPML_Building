from PIL import features
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from spml.models.backbones.vgg import B2_VGG
from spml.models.dt_triton import domain_transform_triton, domain_transform_pytorch
import spml.utils.segsort.common as segsort_common
import spml.utils.general.common as common_utils
from spml.models.predictions.segsort_softmax import *

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Edge_Module(nn.Module):

    def __init__(self, in_fea=[64, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out



def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, kernel_size=11):
        super(CrissCrossAttention, self).__init__()
        self.h_query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0),
                                 stride=(1, 1))
        self.w_query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2),
                                 stride=(1, 1))
        self.h_key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0),
                               stride=(1, 1))
        self.w_key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2),
                               stride=(1, 1))
        self.h_value = nn.Conv2d(in_dim, in_dim, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0),
                                 stride=(1, 1))
        self.w_value = nn.Conv2d(in_dim, in_dim, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2),
                                 stride=(1, 1))
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query_h = self.h_query(x)
        proj_query_w = self.w_query(x)
        proj_query_H = (proj_query_h.permute(0, 3, 1, 2).contiguous()
                        .view(m_batchsize * width, -1, height).permute(0, 2, 1))
        proj_query_W = (proj_query_w.permute(0, 2, 1, 3).contiguous()
                        .view(m_batchsize * height, -1, width).permute(0, 2, 1))
        proj_key_w = self.w_key(x)
        proj_key_h = self.h_key(x)
        proj_key_H = proj_key_h.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key_w.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value_h = self.h_value(x)
        proj_value_w = self.w_value(x)
        proj_value_H = proj_value_h.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value_w.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) +
                    self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class VGG_Contrastive(nn.Module):
    #def __init__(self, channel=32, aspp=True, aspp_blocks=2, DT=True, dt_iter=1, kmeans_num_clusters=12, kmeans_iterations=10, ignore_index=128):
    def __init__(self, config):
        super(VGG_Contrastive, self).__init__()
        channel = config.network.embedding_dim
        self.vgg = B2_VGG()
        self.relu = nn.ReLU(True)
        self.edge_layer = Edge_Module()
        self.use_dt= config.network.use_dt
        self.dt_iter = config.network.dt_iter
        self.kmeans_num_clusters = config.network.kmeans_num_clusters
        self.kmeans_iterations = config.network.kmeans_iterations
        self.label_divisor = config.network.label_divisor
        self.semantic_ignore_index = config.dataset.semantic_ignore_index
        # self.boundary_attn = BoundaryAttnLayer(512, blocks, num_heads=16, init_value=1e-5, heads_range=6, lamb=lamb)
        #self.aspp = _AtrousSpatialPyramidPoolingModule(512, channel, output_stride=16)
        if config.network.aspp:
            self.aspp = nn.Sequential(*[CrissCrossAttention(512) for _ in range(config.network.aspp_blocks)])
        else:
            self.aspp = nn.Identity()
        self.rcab_feat = RCAB(channel * 6)
        self.sal_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(channel * 2)
        self.after_aspp_conv5 = nn.Conv2d(512, channel, kernel_size=1, bias=False)
        self.after_aspp_conv2 = nn.Conv2d(128, channel, kernel_size=1, bias=False)
        self.initial_classification = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, bias=False))
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        '''self.final_classification = nn.Sequential(
            #nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel * 2, out_channels=1, kernel_size=3, padding=1)
        )'''
        self.final_classification = nn.Conv2d(channel * 2, 1, kernel_size=1, bias=False)
        #self.contrastive_loss_model = SegsortSoftmax()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def generate_clusters(self, embeddings,
                            semantic_labels,
                            instance_labels,
                            local_features=None):
        """Perform Spherical KMeans clustering within each image.

        Args:
        embeddings: A a 4-D float tensor of shape
            `[batch_size, channels, height, width]`.
        semantic_labels: A 3-D long tensor of shape
            `[batch_size, height, width]`.
        instance_labels: A 3-D long tensor of shape
            `[batch_size, height, width]`.
        local_features: A 4-D float tensor of shape
            `[batch_size, height, width, channels]`.

        Return:
        A dict with entry `cluster_embedding`, `cluster_embedding_with_loc`,
        `cluster_semantic_label`, `cluster_instance_label`, `cluster_index`
        and `cluster_batch_index`.
        """

        if semantic_labels is not None and instance_labels is not None:
            labels = semantic_labels * self.label_divisor + instance_labels
            ignore_index = labels.max() + 1
            labels = labels.masked_fill(
                semantic_labels == self.semantic_ignore_index,
                ignore_index)
        else:
            labels = None
            ignore_index = None

        # Spherical KMeans clustering.
        (cluster_embeddings,
        cluster_embeddings_with_loc,
        cluster_labels,
        cluster_indices,
        cluster_batch_indices) = (
        segsort_common.segment_by_kmeans(
            embeddings,
            labels,
            self.kmeans_num_clusters,
            local_features=local_features,
            ignore_index=ignore_index,
            iterations=self.kmeans_iterations))

        cluster_semantic_labels = cluster_labels // self.label_divisor
        cluster_instance_labels = cluster_labels % self.label_divisor

        outputs = {
        'cluster_embedding': cluster_embeddings,
        'cluster_embedding_with_loc': cluster_embeddings_with_loc,
        'cluster_semantic_label': cluster_semantic_labels,
        'cluster_instance_label': cluster_instance_labels,
        'cluster_index': cluster_indices,
        'cluster_batch_index': cluster_batch_indices,
        }

        return outputs

    def generate_location_features(self, embeddings):
        n,c,h,w = embeddings.shape
        features = []
        locations = segsort_common.generate_location_features(
            (h,w), embeddings.device, 'float'
        )
        locations -= 0.5
        locations=locations.unsqueeze(0).expand(n,h,w,2)
        features.append(locations)
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, data, targets=None):
        x = data['image']
        x_size = x.size()
        x1 = self.vgg.conv1(x)  ## 352*352*64
        x2 = self.vgg.conv2(x1)  ## 176*176*128
        x3 = self.vgg.conv3(x2)  ## 88*88*256
        x4 = self.vgg.conv4(x3)  ## 44*44*512
        x5 = self.vgg.conv5(x4)  ## 22*22*512
        edge_map = self.edge_layer(x1, x3, x4)
        edge_map_DT = torch.where(edge_map > 0.1, edge_map, torch.zeros_like(edge_map))
        edge_out = torch.sigmoid(edge_map)
        ####
        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            gray_img = cv2.cvtColor(im_arr[i], cv2.COLOR_RGB2GRAY)
            canny[i] = cv2.Canny(gray_img, 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.fuse_canny_edge(cat)
        acts = torch.sigmoid(acts)

        x5 = self.aspp(x5)
        x_conv5 = self.after_aspp_conv5(x5)
        x_conv2 = self.after_aspp_conv2(x2)
        x_conv5_up = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=True)

        feat_fuse = torch.cat([x_conv5_up, x_conv2], 1)

        sal_init = self.initial_classification(feat_fuse)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')

        sal_feature = self.sal_conv(sal_init)
        '''if self.use_dt:
            sal_feature_dt = domain_transform_triton(sal_feature, edge_map_DT, sigma_s=130, sigma_r=0.1, num_iterations=self.dt_iter)
        else:
            sal_feature_dt = sal_feature'''
        edge_feature = self.edge_conv(edge_map)
        sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1))
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        if self.use_dt:
            sal_edge_feature = domain_transform_triton(sal_edge_feature, edge_map_DT, sigma_s=130, sigma_r=0.1, num_iterations=self.dt_iter)
        embedding_feature = sal_edge_feature.clone()
        # 删除这个normalize
        sal_edge_feature = F.normalize(sal_edge_feature, p=2, dim=1)
        sal_ref = self.final_classification(sal_edge_feature)

        # embedding feature for contrastive learning
        if targets is not None:
            local_feature = self.generate_location_features(embedding_feature)
            cluster_embeddings = self.generate_clusters(
                embedding_feature,
                targets['semantic_label'] if targets is not None else None,
                targets['instance_label'] if targets is not None else None,
                local_features=local_feature
            )
            cluster_embeddings['local_feature'] = local_feature
        else:
            cluster_embeddings = {}

        cluster_embeddings['sal_init'] = sal_init
        cluster_embeddings['sal_ref'] = sal_ref
        cluster_embeddings['edge_map'] = edge_map
        cluster_embeddings['embedding'] = embedding_feature
        
        return cluster_embeddings

    def get_params_lr(self):
        """Helper function to adjust learning rate for each sub modules.

        Policy:
        - Backbone (VGG conv3/4/5): base lr (1x) for weights, (2x) for biases (no weight decay).
        - ASPP / aspp-like module: 10x lr for weights, 20x for biases.
        - Newly added heads (edge, saliency, fusion, final classifier, rcab blocks, etc):
            10x lr for weights, 20x for biases.
        """
        ret = []

        # Backbone: use conv3/conv4/conv5 as the trainable "deep" backbone layers
        vgg_backbone_prefixes = ['vgg.conv3', 'vgg.conv4', 'vgg.conv5']
        ret.append({
            'params': [p for p in model_utils.get_params(self, vgg_backbone_prefixes, ['weight'])],
            'lr': 1.0
        })
        ret.append({
            'params': [p for p in model_utils.get_params(self, vgg_backbone_prefixes, ['bias'])],
            'lr': 1,  # 2
            'weight_decay': 0
        })

        # ASPP / aspp-like module (if present) -> higher LR
        ret.append({
            'params': [p for p in model_utils.get_params(self, ['aspp'], ['weight'])],
            'lr': 1  # 10
        })
        ret.append({
            'params': [p for p in model_utils.get_params(self, ['aspp'], ['bias'])],
            'lr': 1, # 20
            'weight_decay': 0
        })

        # Heads / newly added modules: give them higher LR (10x)
        head_prefixes = [
            'edge_layer',
            'edge_conv',
            'sal_conv',
            'rcab_feat',
            'rcab_sal_edge',
            'after_aspp_conv5',
            'after_aspp_conv2',
            'initial_classification',
            'fuse_canny_edge',
            'final_classification'
        ]

        # weights of heads
        ret.append({
            'params': [p for p in model_utils.get_params(self, head_prefixes, ['weight'])],
            'lr': 1.0  # 10
        })
        # biases of heads (no weight decay)
        ret.append({
            'params': [p for p in model_utils.get_params(self, head_prefixes, ['bias'])],
            'lr': 1, # 20
            'weight_decay': 0
        })

        # Fallback: if any parameter wasn't captured above, you can add a final group
        # (optional) — here we don't add it to avoid duplications. If you see "no params"
        # in some groups, you can add a generic group like below:
        # ret.append({'params': [p for p in self.parameters() if p.requires_grad], 'lr': 1.0})

        return ret
    
    