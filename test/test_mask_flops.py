import torch


class MyModel:

    def __init__(self, flops_threshold=None):
        self.flops_threshold = flops_threshold

    def flops_value(self, representation, group_num=1):
        # representation size: (ndevice * batch_size) * vocab_dim
        # group num: how many semantic similar documents representations in one batch
        if self.flops_threshold is None:
            representation = representation.reshape(-1, group_num,
                                                    representation.shape[-1])
            return torch.sum(torch.mean(torch.abs(representation), dim=0)**2)
        else:
            representation = representation.reshape(-1, group_num,
                                                    representation.shape[-1])
            flops_per_token = torch.abs(representation)
            mask = (flops_per_token > self.flops_threshold).float()
            flops_per_average_token = torch.mean(mask * flops_per_token, dim=0) ** 2
            return torch.sum(flops_per_average_token)

# Create an instance of your model with no threshold
model_no_threshold = MyModel()

# Generate a random representation tensor
batch_size = 2
vocab_dim = 4
representation = torch.tensor([[1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

# Calculate the flops value without a threshold
loss_no_threshold = model_no_threshold.flops_value(representation, group_num=2)
print("Loss without threshold:", loss_no_threshold.item())

# Create an instance of your model with a threshold of 0.5
model_with_threshold = MyModel(flops_threshold=0.5)

# Calculate the flops value with a threshold of 0.5
loss_with_threshold = model_with_threshold.flops_value(representation,
                                                       group_num=2)
print("Loss with threshold 0.5:", loss_with_threshold.item())
