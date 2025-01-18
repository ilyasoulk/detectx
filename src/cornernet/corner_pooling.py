class TopLeftPool(nn.Module):
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pool = torch.zeros_like(x)
        for i in range(height):
            for j in range(width):
                pool[:, :, i, j] = torch.max(torch.max(x[:, :, i:, j], dim=2)[0], dim=2)[0]
        return pool

class BottomRightPool(nn.Module):
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pool = torch.zeros_like(x)
        for i in range(height):
            for j in range(width):
                pool[:, :, i, j] = torch.max(torch.max(x[:, :, :i+1, :j+1], dim=2)[0], dim=2)[0]
        return pool
