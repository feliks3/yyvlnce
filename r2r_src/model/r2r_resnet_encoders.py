import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder


class TorchVisionResNet152(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self,
        observation_rgb_size,
        output_size,
        device,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self._n_input_rgb = observation_rgb_size[2]
        obs_size_0 = observation_rgb_size[0]
        obs_size_1 = observation_rgb_size[1]
        linear_layer_input_size += self.resnet_layer_size

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet152(pretrained=True)
        self.cnn.cuda(device)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(
                1, dtype=torch.float32, device=self.device
            )

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        rgb_observations = observations.permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )  # [BATCH x OUTPUT_DIM]
