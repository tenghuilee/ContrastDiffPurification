from tqdm import tqdm
import numpy as np
import math
import torchvision.utils as vutils
import score_sde.sampling as sampling
import score_sde.sde_lib as sde_lib
import guided_diffusion.gaussian_diffusion as gaussian_diffusion
from guided_diffusion.unet import UNetModel
from score_sde.models import utils as mutils
from basic_config_setting import ConfigMain
import unittest
import utils
import torch
from utils import str2bool, get_accuracy, get_image_classifier, load_data
import sys
sys.path.append("..")


def unsqueeze3x(x): return x[..., None, None, None]


class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) /
                               1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars["alpha_bar"][t].sqrt())
                * (xt - unsqueeze3x((1 - scalars["alpha_bar"][t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars["alpha_bar"][t].sqrt() * scalars["beta"][t])
                / ((1 - scalars["alpha_bar"][t]) * scalars["alpha"][t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars["alpha"][t] - scalars["alpha_bar"][t])
                / ((1 - scalars["alpha_bar"][t]) * scalars["alpha"][t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) /
                            alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return dict([(k, v.float()) for (k, v) in all_scalars.items()])

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.
        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.
        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar /
            torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor(
                    [i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"]
                                        [current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 -
                                unsqueeze3x(
                                    scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final


def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.
    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.
    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    while num_samples < N:
        if xT is None:
            xT = (
                torch.randn(batch_size, num_channels, image_size, image_size)
                .float().cuda()
            )
        y = None
        gen_images = diffusion.sample_from_reverse_process(
            model, xT, sampling_steps, {"y": y}, ddim=True,
        )
        samples.append(gen_images.detach().cpu().numpy())
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None)


def UNet(
    image_size,
    in_channels=3,
    out_channels=3,
    base_width=64,
    num_classes=None,
):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    if image_size == 28:
        attention_resolutions = "28,14,7"
    else:
        attention_resolutions = "32,16,8"
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=3,
        attention_resolutions=tuple(attention_ds),
        dropout=0.1,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )


class MainTest(unittest.TestCase):
    def test_cifar100(self):
        model = utils.get_image_classifier("cifar100-wideresnet-28-10").cuda()

        # print(model)

        class _args:
            domain: str = "cifar100"
            num_sub: int = 16
            data_seed: int = 1

        x_val, y_val = utils.load_data(_args(), 128)

        predict = model.forward(x_val.cuda())

        acc = torch.sum(torch.argmax(predict, 1) ==
                        y_val.cuda()) / y_val.shape[0]

        print(acc)

    # def test_cifar100_score_sde(self):
    #     config = ConfigMain.from_yaml("./configs/cifar100.yml")

    #     model = mutils.create_model(config).eval()

    def test_gtsrb_resnet50(self):
        model = utils.get_image_classifier("gtsrb-resnet50-10").cuda()

        # print(model)

        class _args:
            domain: str = "gtsrb"
            num_sub: int = 16
            data_seed: int = 1

        x_val, y_val = utils.load_data(_args(), 128)

        predict = model.forward(x_val.cuda())

        acc = torch.sum(torch.argmax(predict, 1) ==
                        y_val.cuda()) / y_val.shape[0]

        print(acc)

    def test_gtsrb_unet(self):
        model = UNet(
            image_size=32,
            num_classes=43,
        )

        model.load_state_dict(torch.load(
            "./pretrained/gtsrb/UNet_gtsrb-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt",
            map_location=torch.device("cpu"),
        ))

        model.num_classes = None  # ignore classes
        model.eval().cuda()

        diffusion = gaussian_diffusion.GaussianDiffusion(
            betas=gaussian_diffusion.get_named_beta_schedule("cosine", 1000),
            model_mean_type=gaussian_diffusion.ModelMeanType.EPSILON,
            model_var_type=gaussian_diffusion.ModelVarType.FIXED_LARGE,
            loss_type=gaussian_diffusion.LossType.MSE,  # ignore
            rescale_timesteps=False,
        )

        out = diffusion.ddim_sample_loop(
            model,
            (64, 3, 32, 32),
            clip_denoised=True,
            # model_kwargs={
            #     "y": torch.randint(43, (64, ), dtype=torch.int64).cuda(),
            # }
        )  # type: torch.Tensor
        out = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)
        # out = (out - out.min())
        # out = out / out.max()
        # print(out.max(), out.min())

        # diffusion = GuassianDiffusion()

        # out = sample_N_images(1, model, diffusion, None)

        vutils.save_image(out, "./test_gtsrb_unet_nolabel.png")

        print("eval unet pass")
