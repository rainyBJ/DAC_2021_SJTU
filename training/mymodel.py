import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_dorefa import *


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)   # 下采样步长

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])  # yv和xv的行数与ny的元素个数相等，yv和xv的列数与nx的元素个数相等。 yv各行元素相同，xv各列元素相同
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.no = 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
    def forward(self, p, img_size):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # [bs, #anchors, H, W, #preds]

        if self.training:  # training mode
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride  # 原始像素尺度

            
            torch.sigmoid_(io[..., 4:])
            
            
            return io.view(bs, -1, self.no), p  # [bs, n(#anchors*h*w), 6]


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
        x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
        x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
        x = x.view([B, hs*ws*C, H//hs, W//ws])
        return x


class UltraNet_Bypass(nn.Module):
    def __init__(self):
        super(UltraNet_Bypass, self).__init__()
        W_BIT = 4
        A_BIT = 4
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)
        self.reorg = ReorgLayer(stride=2)

        self.layers_p1 = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2)
        )

        self.layers_p2 = nn.Sequential(
            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT)
        )

        self.layers_p3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),
        )

        self.layers_p4 = nn.Sequential(
            conv2d_q(320, 64, kernel_size=3, stride=1, padding=1, bias=False),   # cat p2--64→64*4 + 64
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)
        )
        self.yololayer = YOLOLayer([[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])  # 6 anchors x 6 predictions
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.layers_p1(x)  # [bs, c, h, w]
        x_p2 = self.layers_p2(x_p1)
        x_p2_reorg = self.reorg(x_p2)
        x_p3 = self.layers_p3(x_p2)
        x_p4_in = torch.cat([x_p2_reorg, x_p3], 1)
        x_p4 = self.layers_p4(x_p4_in)

        x = self.yololayer(x_p4, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p  # concat at #predictions
        return x


class UltraNet_Bypass_k(nn.Module):
    def __init__(self):
        super(UltraNet_Bypass_k, self).__init__()
        W_BIT = 4
        A_BIT = 4
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)
        self.reorg = ReorgLayer(stride=2)

        self.layers_p1 = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2)
        )

        self.layers_p2 = nn.Sequential(
            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT)
        )

        self.layers_p3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),
        )

        self.layers_p4 = nn.Sequential(
            conv2d_q(320, 64, kernel_size=3, stride=1, padding=1, bias=False),   # cat p2--64→64*4 + 64
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)  # 最后一层，使用 conv 实现 linear 的效果
        )
        self.yololayer = YOLOLayer([[4, 11], [8, 28], [12, 48], [20, 33], [18, 69], [35, 91]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.layers_p1(x)
        x_p2 = self.layers_p2(x_p1)
        x_p2_reorg = self.reorg(x_p2)
        x_p3 = self.layers_p3(x_p2)
        x_p4_in = torch.cat([x_p2_reorg, x_p3], 1)
        x_p4 = self.layers_p4(x_p4_in)

        x = self.yololayer(x_p4, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x


class SkyNet(nn.Module):
    def __init__(self):
        super(SkyNet, self).__init__()
    
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        
        self.reorg = ReorgLayer(stride=2)
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.model_p1 = nn.Sequential(
            conv_dw( 3,  48, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),   #dw3
        )    
        self.model_p2 = nn.Sequential(    
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),   #dw4
            conv_dw(384, 512, 1),   #dw5
        )
        self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
            conv_dw(1280, 96, 1),
            nn.Conv2d(96, 18, 1, 1,bias=False),
        )
        self.yololayer = YOLOLayer([[10,14],  [23,27],  [37,58]])

        self.yolo_layers = [self.yololayer]
        # self._initialize_weights()
    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        x = self.model_p3(x_p3_in)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x


class UltraNet(nn.Module):
    def __init__(self):
        super(UltraNet, self).__init__()
        W_BIT = 8
        A_BIT = 8
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)

        self.layers = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x
