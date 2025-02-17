import torch
import torch.nn.functional as F
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 如果 A4000 不是 GPU 0，可修改为 "1"


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=8,
            spline_order=4,
            scale_noise=0.5,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.01,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Initialize weights
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # Initialize parameters
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation() if isinstance(base_activation, type) else base_activation
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() >= 2 and x.size(-1) == self.in_features

        # Reshape x to (batch_size, 1, in_features, 1)
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, in_features, 1)

        # Use self.grid and reshape to (1, 1, in_features, grid_size + 2 * spline_order + 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, in_features, grid_size + 2 * spline_order + 1)

        # Initialize bases
        bases = ((x >= grid[:, :, :, :-1]) & (x < grid[:, :, :, 1:])).to(
            x.dtype)  # (batch_size, 1, in_features, grid_size + 2 * spline_order)

        # Compute B-splines using Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            # Compute the left term
            left = (x - grid[:, :, :, :-(k + 1)]) / (grid[:, :, :, k:-1] - grid[:, :, :, :-(k + 1)]) * bases[:, :, :,
                                                                                                       :-1]

            # Compute the right term
            right = (grid[:, :, :, k + 1:] - x) / (grid[:, :, :, k + 1:] - grid[:, :, :, 1:(-k)]) * bases[:, :, :, 1:]

            # Combine the terms
            bases = left + right

        # Trim the bases to the correct size
        num_elements_to_retain = self.grid_size + self.spline_order
        bases = bases[:, :, :, :num_elements_to_retain]  # Retain only the valid bases

        # Remove the extra dimension
        bases = bases.squeeze(1)  # (batch_size, in_features, grid_size + spline_order)


        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order), \
            f"Expected shape: {(x.size(0), self.in_features, self.grid_size + self.spline_order)}, got {bases.size()}"

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() >= 2 and x.size(-1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # Shape: (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # Shape: (in_features, batch_size, out_features)


        # Solve the linear system A * X = B
        solution = torch.linalg.lstsq(A, B).solution  # Shape: (in_features, grid_size + spline_order, out_features)

        # Permute the dimensions to get the correct shape
        result = solution.permute(2, 0, 1)  # Shape: (out_features, in_features, grid_size + spline_order)


        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        ), f"Expected shape: {(self.out_features, self.in_features, self.grid_size + self.spline_order)}, got {result.size()}"

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() >= 2 and x.size(-1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() >= 2 and x.size(-1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coefficients = self.scaled_spline_weight
        orig_coefficients = orig_coefficients.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coefficients)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=0.1, regularize_entropy=0.1):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())

        total_regularization_loss = (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=8,
            spline_order=4,
            scale_noise=0.5,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.01,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Handle hidden layers and flatten for higher dimensions
        self.layers_hidden = layers_hidden

        self.layers = torch.nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            if i == 0 and isinstance(in_features, (list, tuple)):
                # For the first layer, calculate the flattened input dimension
                in_features = math.prod(in_features)

            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        前向传播逻辑。
        如果 update_grid 为 True，则更新网格点。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, ..., in_features)。
            update_grid (bool): 是否在前向传播时更新网格点。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, ..., out_features)。
        """
        # 记录输入形状
        original_shape = x.shape
        # print("Input tensor shape before flattening:", original_shape)

        # 如果输入是高维张量，先将其展平为二维张量
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten spatial dimensions

        # Ensure the input tensor has the correct shape
        expected_features = self.layers[0].in_features

        if x.size(-1) != expected_features:
            raise ValueError(f"Expected input features: {expected_features}, but got {x.size(-1)}")

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算所有层的正则化损失。

        Args:
            regularize_activation (float): 激活正则化权重。
            regularize_entropy (float): 熵正则化权重。

        Returns:
            torch.Tensor: 正则化损失值。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# 测试代码
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 计算输入张量的展平大小
#     flattened_size = 128 * 16 * 16
#     # 定义测试输入
#     batch_size = 4
#     input_features = [128, 16, 16]  # 4D input (batch_size, height, width, channels)
#     output_features = 128
#     hidden_layers = [input_features[0] * input_features[1] * input_features[2], 32, 64, 10]  # 减小隐藏层维度
#     # hidden_layers = [flattened_size, 61, 128, output_features]  # 修改 hidden_layers，第一个是展平后的大小
#
#     # 创建 KAN 模型
#     model = KAN(
#         layers_hidden=hidden_layers,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.2,
#         base_activation=torch.nn.SiLU,
#         grid_eps=0.01,
#         grid_range=[-1, 1],
#     ).to(device)
#
#     # 打印模型结构
#     # print(model)
#
#     # 创建随机输入数据
#     x = torch.rand(batch_size, *input_features).to(device)
#
#     # 如果输入张量的形状不匹配，调整其形状
#     if x.numel() // x.size(0) != hidden_layers[0]:
#         # 使用平均池化调整形状
#         x = F.avg_pool3d(x, kernel_size=4).to(device)  # 调整 kernel_size 以匹配期望的输入尺寸
#
#     # 测试前向传播
#     output = model(x)
#     # print("Output shape:", output.shape)
#
#     # 测试正则化损失
#     reg_loss = model.regularization_loss()
#     # print("Regularization loss:", reg_loss.item())