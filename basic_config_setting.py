from dataclasses import dataclass, field
import yaml


@dataclass
class DataConfig:
    dataset: str = "CIFAR100"
    data_path: str = ""
    is_load_training_set: bool = False
    category: str = "cifar100"
    attribute: str = ""
    image_size: int = 32
    num_channels: int = 3
    random_flip: bool = True
    centered: bool = True
    uniform_dequantization: bool = False


@dataclass
class ModelConfig:
    sigma_min: float = 0.01
    sigma_max: float = 50
    num_scales: int = 1000
    beta_min: float = 0.1
    beta_max: float = 20.
    dropout: float = 0.1

    name: str = 'ncsnpp'
    scale_by_sigma: bool = False
    ema_rate: float = 0.9999
    normalization: str = 'GroupNorm'
    nonlinearity: str = 'swish'
    nf: int = 128
    ch_mult: list = field(default_factory=lambda: [1, 2, 2, 2])
    num_res_blocks: int = 8
    attn_resolutions: list = field(default_factory=lambda: [16])
    resamp_with_conv: bool = True
    conditional: bool = True
    fir: bool = False
    fir_kernel: list = field(default_factory=lambda: [1, 3, 3, 1])
    skip_rescale: bool = True
    resblock_type: str = 'biggan'
    progressive: str = 'none'
    progressive_input: str = 'none'
    progressive_combine: str = 'sum'
    attention_type: str = 'ddpm'
    init_scale: float = 0.
    embedding_type: str = 'positional'
    fourier_scale: int = 16
    conv_size: int = 3


@dataclass
class TrainingConfig:
    sde: str = 'vpsde'
    continuous: bool = True
    reduce_mean: bool = True
    likelihood_weighting: bool = False
    snapshot_sampling: bool = True
    n_iters: int = 950001
    batch_size: int = 128


@dataclass
class OptimConfig:
    weight_decay: float = 0
    optimizer: str = 'Adam'
    lr: float = 0.0002  # 2e-4
    beta1: float = 0.9
    eps: float = 0.00000001  # 1e-8
    warmup: int = 5000
    grad_clip: float = 1.


@dataclass
class SamplingConfig:
    n_steps_each: int = 1
    noise_removal: bool = True
    probability_flow: bool = False
    snr: float = 0.16

    method: str = 'pc'
    predictor:  str = 'euler_maruyama'
    corrector:  str = 'none'


@dataclass
class ConfigMain:
    data: DataConfig = field(default_factory=lambda: DataConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    optim: OptimConfig = field(default_factory=lambda: OptimConfig())
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())


    @staticmethod
    def from_yaml(yaml_file: str):
        with open(yaml_file, 'r') as f:
            cfg = yaml.safe_load(f)
        return ConfigMain(
            data=DataConfig(**cfg['data']),
            model=ModelConfig(**cfg['model']),
            training=TrainingConfig(**cfg['training']),
            optim=OptimConfig(**cfg['optim']),
            sampling=SamplingConfig(**cfg['sampling']),
        )
