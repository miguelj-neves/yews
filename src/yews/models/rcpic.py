import torch.nn as nn
from torch import load

from .utils import load_state_dict_from_url

__all__ = ["RCpicV1", "rcpic_v1", 'cpicv1_dropout', 'CpicDropout']

#model_urls = {
#    "cpic_v1": "https://www.dropbox.com/s/ckb4glf35agi9xa/cpic_v1_wenchuan-bdd92da2.pth?dl=1",
#    "cpic_v2": "https://www.dropbox.com/s/kyiuprnn8014fs5/cpic_v2_wenchuan-ee92060a.pth?dl=1",
#    "cpic_v3": "",
#}


class RCpicV1(nn.Module):
    def __init__(self):
        super().__init__()
        # 4000 -> 2000
        self.layer1  = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2),
        )
        # 2000 -> 1024
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=26, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2),
        )

        # 1024 -> 512
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2),
        )

        # 512 -> 256
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 256 -> 128
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 128 -> 64
        self.layer6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 64 -> 32
        self.layer7 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 32 -> 16
        self.layer8 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 16 -> 8
        self.layer9 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 8 -> 4
        self.layer10 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 4 -> 2
        self.layer11 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 2 -> 1
        self.layer12 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Linear(64 * 1, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
class CpicDropout(nn.Module):
    def __init__(self,dropout=0.08):
        super().__init__()
        if isinstance(dropout, float):
            self.dropout = dropout
        else:
            raise ValueError("Dropout probability needs to be a float.")
        # Add dropout at each layer
        
        # 2000 -> 1024
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=26, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 1024 -> 512
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 512 -> 256
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 256 -> 128
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 128 -> 64
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 64 -> 32
        self.layer6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 32 -> 16
        self.layer7 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 16 -> 8
        self.layer8 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 8 -> 4
        self.layer9 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 4 -> 2
        self.layer10 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(2)
        )

        # 2 -> 1
        self.layer11 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(p=0.05),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Linear(64 * 1, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def rcpic_v1(pretrained=False, progress=True, path=None, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = RCpicV1(**kwargs)
    if pretrained:
        if path != None:
            model.load_state_dict(load(path)['model'])
    return model





def cpic(pretrained=False, progress=True, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return cpic_v1(pretrained, progress, **kwargs)

def cpicv1_dropout(pretrained=False, progress=True, dropout=0.3, path=None, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = CpicDropout(dropout=dropout, **kwargs)
    if pretrained:
        if path != None:
            model.load_state_dict(load(path)['model'])
        else:
            state_dict = load_state_dict_from_url(model_urls['cpicv1_dropout'],
                                                  progress=progress)
            model.load_state_dict(state_dict)
    return model
