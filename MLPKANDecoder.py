import torch
import torch.nn as nn
from model.CNNWITHKAN.KAN import KAN


class MLPKANDecoder(nn.Module):

    def __init__(self, l1_channels, l2_channels, l3_channels, l4_channels, class_num):
        super().__init__()

        # self.target_size = (16, 16)

        self.dec_layer1 = nn.Sequential(
            nn.Conv2d(l1_channels, 64, kernel_size=1),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)  # 确保所有尺寸一致
        )

        self.dec_layer2 = nn.Sequential(
            nn.Conv2d(l2_channels, 64, kernel_size=1),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)
        )

        self.dec_layer3 = nn.Sequential(
            nn.Conv2d(l3_channels, 64, kernel_size=1),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)
        )

        self.dec_layer4 = nn.Sequential(
            nn.Conv2d(l4_channels, 64, kernel_size=1),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)
        )

        # self.dec_layer1 = nn.Sequential(
        #     nn.Conv2d(l1_channels, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=1/8, mode="bilinear", align_corners=True)
        # )
        # self.dec_layer2 = nn.Sequential(
        #     nn.Conv2d(l2_channels, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=1/4, mode="bilinear", align_corners=True)
        # )
        # self.dec_layer3 = nn.Sequential(
        #     nn.Conv2d(l3_channels, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=1/2, mode="bilinear", align_corners=True)
        # )
        # self.dec_layer4 = nn.Sequential(
        #     nn.Conv2d(l4_channels, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
        # )
        # self.linear1 = nn.Conv2d(128 * 4, 128, kernel_size=1)
        # self.linear2 = nn.Conv2d(128, class_num, kernel_size=1)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(128)
        #
        # self.reduce = nn.Conv2d(128, 32, kernel_size=1)  # 降维到 32
        # self.expand = nn.Conv2d(32, 128, kernel_size=1)  # 还原 128

        # 先降维，再展平输入 KAN
        self.reduce = nn.Conv2d(64 * 4, 64, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mlp_layer = nn.Linear(512, 128)
        # self .dropout_mlp = nn.Dropout(p=0.2)
        # self.linear1 = nn.Conv2d(128 * 4, 128, kernel_size=1)
        # self.linear2 = nn.Conv2d(128, class_num, kernel_size=1)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(128)

        flatten_dim = 64 * 8 * 8
        self.kan_layer = KAN(
            layers_hidden=[flatten_dim, 1024, 1024, flatten_dim],  # KAN 处理最后的映射
            grid_size=3,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,  # ✅ 正确传递
            grid_eps=0.01,
            grid_range=[-1, 1],
        )
        # 最终分类层
        self.classifier = nn.Conv2d(64, class_num, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.SiLU(inplace=True)
        self.final_upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)

        # # 这里先保持 64通道 => (class_num)
        # self.final_conv = nn.Conv2d(64, class_num, kernel_size=1)
        # self.upsample = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True)
        #
        # self.SiLU = nn.SiLU()

        # self.expand = nn.Conv2d(32, 64, kernel_size=1)
        # self.bn = nn.BatchNorm1d(128)

        # 最终分类层
        # self.linear1 = nn.Conv2d(64, 64, kernel_size=1)
        # self.linear2 = nn.Conv2d(64, class_num, kernel_size=1)
        # self.relu = nn.ReLU()

    def forward(self, x1, x2, x3, x4):
        # batch_size = x1.size(0)
        x1 = self.dec_layer1(x1)
        x2 = self.dec_layer2(x2)
        x3 = self.dec_layer3(x3)
        x4 = self.dec_layer4(x4)

        # assert x1.shape == x2.shape == x3.shape == x4.shape

        # 拼接通道
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.reduce(x)
        # 适配 KAN
        x = x.contiguous()
        batch_size, c, h, w = x.shape
        x_flat = x.view(batch_size, -1)
        x_kan = self.kan_layer(x_flat)
        x_reshape = x_kan.view(batch_size, 64, h, w)
        x_reshape = self.dropout(x_reshape)
        x_reshape = self.act(self.bn(x_reshape))

        out = self.classifier(x_reshape)
        out = self.final_upsample(out)  # => (N, class_num, 128,128)

        # x = x.view(batch_size, -1)
        # x = self.kan_layer(x)
        # x = x.view(batch_size, 64, h, w)  # 重新 reshape 回二维
        #
        # # x = self.expand(x)  # 还原回 64 维度
        # x = self.dropout(x)  # Dropout 防止过拟合
        #
        # # x = self.linear1(x)
        # x = self.SiLU(x)
        # x = self.bn(x)
        # # x = self.linear2(x)
        # # final conv => (N,class_num,16,16)
        # x = self.final_conv(x)
        #
        # # x = self.upsample(x)
        return out


        # x = self.expand(x)  # 还原回 64 维度
        # x = self.dropout(x)  # Dropout 防止过拟合
        # x = self.linear1(x)
        # x = self.SiLU(x)
        # x = self.bn(x)
        # x = self.linear2(x)
        # x = self.upsample(x)
        #
        # return x

#         self.kan_layer1 = KAN(
#             layers_hidden=[64 * 16 * 16, 32 * 8 * 8],  # Input size: 64 * 16 * 16
#             grid_size=5,
#             spline_order=3,
#             scale_noise=0.1,
#             scale_base=1.0,
#             scale_spline=1.0,
#             base_activation=torch.nn.SiLU,
#             grid_eps=0.02,
#             grid_range=[-1, 1],
#         )
#         self.kan_layer2 = KAN(
#             layers_hidden=[128 * 8 * 8, 32 * 8 * 8],  # Adjusted to match reshape
#             grid_size=5,
#             spline_order=3,
#             scale_noise=0.1,
#             scale_base=1.0,
#             scale_spline=1.0,
#             base_activation=torch.nn.SiLU,
#             grid_eps=0.02,
#             grid_range=[-1, 1],
#         )
#         self.kan_layer3 = KAN(
#             layers_hidden=[128 * 4 * 4, 32 * 4 * 4],  # Adjusted to match reshape
#             grid_size=5,
#             spline_order=3,
#             scale_noise=0.1,
#             scale_base=1.0,
#             scale_spline=1.0,
#             base_activation=torch.nn.SiLU,
#             grid_eps=0.02,
#             grid_range=[-1, 1],
#         )
#         self.kan_layer4 = KAN(
#             layers_hidden=[128 * 2 * 2, 32 * 2 * 2],  # Adjusted to match reshape
#             grid_size=5,
#             spline_order=3,
#             scale_noise=0.1,
#             scale_base=1.0,
#             scale_spline=1.0,
#             base_activation=torch.nn.SiLU,
#             grid_eps=0.02,
#             grid_range=[-1, 1],
#         )
#
#         # Final linear layers
#         self.linear1 = nn.Conv2d(32 * 4, 64, kernel_size=1)
#         self.linear2 = nn.Conv2d(64, class_num, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(64)
#
#     def forward(self, x1, x2, x3, x4):
#         batch_size = x1.size(0)
#
#         # Downsample the input to reduce computational complexity
#         x1 = self.dec_layer1(x1)  # 输出形状: (batch_size, 256, H1, W1)
#         x2 = self.dec_layer2(x2)  # 输出形状: (batch_size, 256, H2, W2)
#         x3 = self.dec_layer3(x3)  # 输出形状: (batch_size, 256, H3, W3)
#         x4 = self.dec_layer4(x4)  # 输出形状: (batch_size, 256, H4, W4)
#
#         # Print shapes for debugging
#         print(f"x1 shape: {x1.shape}")
#         print(f"x2 shape: {x2.shape}")
#         print(f"x3 shape: {x3.shape}")
#         print(f"x4 shape: {x4.shape}")
#
#         # KAN 计算（降维后输入）
#         x1 = x1.view(batch_size, -1)  # (batch_size, 32 * 16 * 16)
#         x1 = self.kan_layer1(x1)
#         x1 = x1.view(batch_size, 32, 16, 16)
#
#         x2 = x2.view(batch_size, -1)  # (batch_size, 32 * 8 * 8)
#         x2 = self.kan_layer2(x2)
#         x2 = x2.view(batch_size, 32, 8, 8)
#
#         x3 = x3.view(batch_size, -1)  # (batch_size, 32 * 4 * 4)
#         x3 = self.kan_layer3(x3)
#         x3 = x3.view(batch_size, 32, 4, 4)
#
#         x4 = x4.view(batch_size, -1)  # (batch_size, 32 * 2 * 2)
#         x4 = self.kan_layer4(x4)
#         x4 = x4.view(batch_size, 32, 2, 2)
#
#         # # 还原 128 维
#         # x1 = self.expand(x1)
#         # x2 = self.expand(x2)
#         # x3 = self.expand(x3)
#         # x4 = self.expand(x4)
#
#         # 拼接特征图
#         x = torch.cat([x1, x2, x3, x4], dim=1)  # (batch_size, 128 * 4, 16, 16)
#
#         # 通过最终分类层
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.bn(x)
#         x = self.linear2(x)
#
#         return x
#
#         # # Flatten the feature maps and pass through KAN
#         # x1 = x1.reshape(batch_size, -1)  # Flatten to (batch_size, 64 * 16 * 16)
#         # self.reduce1 = nn.Conv2d(128, 32, kernel_size=1)  # 降维到 32
#         # x1 = self.kan_layer1(x1)  # Pass through KAN
#         # self.expand1 = nn.Conv2d(32, 128, kernel_size=1)  # 再升维回来
#         # x1 = x1.reshape(batch_size, 128, 16, 16)  # Reshape back
#         #
#         # x2 = x2.reshape(batch_size, -1)  # Flatten to (batch_size, 64 * 8 * 8)
#         # x2 = self.kan_layer2(x2)  # Pass through KAN
#         # x2 = x2.reshape(batch_size, 128, 8, 8)  # Reshape back
#         #
#         # x3 = x3.reshape(batch_size, -1)  # Flatten to (batch_size, 64 * 4 * 4)
#         # x3 = self.kan_layer3(x3)  # Pass through KAN
#         # x3 = x3.reshape(batch_size, 128, 4, 4)  # Reshape back
#         #
#         # x4 = x4.reshape(batch_size, -1)  # Flatten to (batch_size, 64 * 2 * 2)
#         # x4 = self.kan_layer4(x4)  # Pass through KAN
#         # x4 = x4.reshape(batch_size, 128, 2, 2)  # Reshape back
#         #
#         # # Concatenate feature maps
#         # x = torch.cat([x1, x2, x3, x4], dim=1)  # Output shape: (batch_size, 64 * 4, 16, 16)
#         #
#         # # Final linear layers
#         # x = self.linear1(x)
#         # x = self.relu(x)
#         # x = self.bn(x)
#         # x = self.linear2(x)
#         #
#         # return x
#
# # Test code

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Define test input
#     batch_size = 2
#     input_channels = 128
#     input_shape = (input_channels, 128, 128)  # Example input shape
#     class_num = 10
#
#     # Create model
#     model = AllMLPDecoder(
#         l1_channels=input_channels,
#         l2_channels=input_channels,
#         l3_channels=input_channels,
#         l4_channels=input_channels,
#         class_num=class_num
#     ).to(device)
#
#     # Create random input data
#     x1 = torch.rand(batch_size, input_channels, 128, 128).to(device)
#     x2 = torch.rand(batch_size, input_channels, 64, 64).to(device)
#     x3 = torch.rand(batch_size, input_channels, 32, 32).to(device)
#     x4 = torch.rand(batch_size, input_channels, 16, 16).to(device)
#
#     # Test forward pass
#     output = model(x1, x2, x3, x4)
#     print("Output shape:", output.shape)