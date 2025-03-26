import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
import datasets as hfds
import huggingface_hub as hfhub
import torchxrayvision as xrv


class MaskedImageClassifier(nn.Module):
    """
    Allows one to pass both an image x and attribution alpha to the classifier.
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        image_size: tuple[int, int] = (224, 224),
        grid_size: tuple[int, int] = (14, 14)
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.image_size = image_size
        self.grid_size = grid_size
        self.scale_factor = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None
    ):
        """
        x: (bsz, C, H, W)
        alpha: (bsz, *grid_size) or (bsz, grid_size[0] * grid_size[1]).
        """
        # Shape-check x
        bsz, C, H, W = x.shape
        assert (H, W) == self.image_size

        gH, gW = self.grid_size
        assert (H, W) == (gH * self.scale_factor[0], gW * self.scale_factor[1])

        # Make alpha if it does not exist
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        # Mask x (bsz, C, H, W) with an up-scaled alpha (bsz, 1, H, W)
        alpha = F.interpolate(alpha.view(bsz, 1, gH, gW).float(), scale_factor=self.scale_factor)
        out = self.base_classifier(x * alpha)

        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out.keys():
            return out["logits"]
        else:
            return out


class SmoothMaskedImageClassifier(nn.Module):
    """
    Averaged evaluation of an alpha-masked image, where each bit in alpha is kept with probability lambda_.
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        num_samples: int = 64,
        image_size: tuple[int, int] = (224, 224),
        grid_size: tuple[int, int] = (14, 14),
        avg_logits: float = False
    ):
        super().__init__()
        self.masked_image_classifier = MaskedImageClassifier(base_classifier, image_size, grid_size)
        self.mask_dim = grid_size[0] * grid_size[1]
        self.avg_logits = avg_logits

        # If we're close to 1.0, don't bother
        if abs(1 - lambda_) < 1e-4:
            self.lambda_ = 1.0
            self.num_samples =  1
        else:
            self.lambda_ = lambda_
            self.num_samples = num_samples

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None,
    ):
        bsz, C, H, W = x.shape

        # Add noise if applicable
        keep_mask = torch.rand(bsz, self.num_samples, self.mask_dim, device=x.device) <= self.lambda_
        if alpha is None:
            alpha = torch.ones(bsz, self.mask_dim, device=x.device)
        alpha = alpha.view(bsz, 1, self.mask_dim) * keep_mask

        # Make copies of x and pass them through the classifier in a batched mode
        xx = x.view(bsz, 1, C, H, W).repeat(1, self.num_samples, 1, 1, 1)
        y = self.masked_image_classifier(xx.flatten(0,1), alpha.flatten(0,1)).view(bsz, self.num_samples, -1)
        if self.avg_logits:
            return y.mean(dim=1)
        else:
            return F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)).float().mean(dim=1)


def discretized_mus_masks(
    dim: int,
    lambda_: float,
    quant: int,
    v_seed: torch.FloatTensor | None = None,
    device: str = "cpu",
    return_dict: bool = False,
):
    """
    Sample the discretized MuS noise used by certified classifiers.

    Args:
        dim: The dimension.
        lambda_: The drop prob (and Lipschitz const). Should be a multiple of 1/q.
        quant: The quantization parameter (q).
        v_seed: The seed noise.

    Returns:
        A 0/1-valued mask of shape (q, dim).
    """
    q = quant
    lambda_ = int(lambda_ * q) / q

    if v_seed is None:
        v_seed = torch.randint(0, q, (1, dim), device=device) / q

    s_base = ((torch.arange(q, device=device) + 0.5) / q).view(q,1)
    t = (v_seed + s_base).remainder(1.0) # (q, dim)
    s = (t < lambda_).long()
    if return_dict:
        return {"mask": s, "pre_mask": t, "v_seed": v_seed}
    else:
        return s # (q, dim)


class CertifiedMuSImageClassifier(nn.Module):
    """
    The certified variant of MuS for masked image classifiers, with optional masking
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        quant: int = 64,
        image_size = (224, 224),
        grid_size = (14, 14)
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.image_size = image_size
        self.q = quant
        self.lambda_ = int(lambda_ * self.q) / self.q
        self.grid_size = grid_size
        self.scale_factor = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

    
    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None,
        return_all: bool = False
    ):
        # Shape-check x
        bsz, C, H, W = x.shape
        assert (H, W) == self.image_size

        gH, gW = self.grid_size
        assert (H, W) == (gH * self.scale_factor[0], gW * self.scale_factor[1])

        # Make alpha if it does not exist
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        q, mask_dim = self.q, gH * gW
        alpha = alpha.view(bsz, mask_dim)
        all_ys = ()
        for x_, a_ in zip(x, alpha):
            mus_masks = discretized_mus_masks(mask_dim, self.lambda_, q, device=x.device) # (q, mask_dim)
            a_masked = a_.view(1, mask_dim) * mus_masks.view(q, mask_dim) # (q, mask_dim)

            # Multiply x (C, H, W) with an up-scaled a_masked (q, 1, H, W)
            a_masked = F.interpolate(a_masked.view(q, 1, gH, gW).float(), scale_factor=self.scale_factor)
            y = self.base_classifier(x_.view(1, C, H, W) * a_masked) # (q, num_classes)

            # Extract logits as necessary
            if hasattr(y, "logits"):
                y = y.logits
            elif isinstance(y, dict) and "logits" in y.keys():
                y = y["logits"]

            # Convert to one-hot and vote
            y = F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)) # (q, num_classes)
            avg_y = y.float().mean(dim=0) # (num_classes)
            all_ys += (avg_y,)

        all_ys = torch.stack(all_ys) # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * self.lambda_)

        return {
            "logits": all_ys, # (bsz, num_classes)
            "cert_rs": cert_rs # (bsz,)
        }


class MaskedTextClassifier(nn.Module):
    """
    Allows one to pass either an inputs_embeds or inputs_ids, attention mask, and attribution alpha to the classifier.
    """
    def __init__(
        self,
        base_classifier: nn.Module
    ):
        super().__init__()
        self.base_classifier = base_classifier
        # Assume that the classifier comes with these
        self.embed_fn = base_classifier.get_input_embeddings()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ):
        # Exactly one can be present
        assert (input_ids is None) ^ (inputs_embeds is None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_fn(input_ids.to(self.device))

        bsz, L, _ = inputs_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs_embeds.device).long()

        if alpha is None:
            alpha = torch.ones_like(attention_mask)

        attention_mask = (attention_mask * alpha).long()
        out = self.base_classifier(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out.keys():
            return out["logits"]
        else:
            return out


class SmoothMaskedTextClassifier(nn.Module):
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        num_samples: int,
        avg_logits: float = False
    ):
        super().__init__()
        self.masked_text_classifier = MaskedTextClassifier(base_classifier)
        self.avg_logits = avg_logits

        # If we're close to 1.0, don't bother
        if abs(1 - lambda_) < 1e-4:
            self.lambda_ = 1.0
            self.num_samples =  1
        else:
            self.lambda_ = lambda_
            self.num_samples = num_samples

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ):
        assert (input_ids is None) ^ (inputs_embeds is None)

        # Just need something to reference the shape
        inputs = input_ids if inputs_embeds is None else inputs_embeds
        bsz, L = inputs.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs.device)

        # Masks are of size L
        keep_mask = torch.rand(bsz, self.num_samples, L, device=inputs.device) <= self.lambda_
        if alpha is None:
            alpha = torch.ones_like(attention_mask)
        alpha = alpha.view(bsz, 1, L) * keep_mask # Mask the alpha

        if input_ids is not None:
            xx = input_ids.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            mm = attention_mask.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            y = self.masked_text_classifier(
                input_ids = xx.flatten(0,1),
                attention_mask = mm.flatten(0,1),
                alpha = alpha.flatten(0,1)
            ).view(bsz, self.num_samples, -1)

        else:
            xx = inputs_embeds.view(bsz, 1, L, -1).repeat(1, self.num_samples, 1, 1)
            mm = attention_mask.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            y = self.masked_text_classifier(
                inputs_embeds = xx.flatten(0,1),
                attention_mask = mm.flatten(0,1),
                alpha = alpha.flatten(0,1)
            ).view(bsz, self.num_samples, -1)

        if self.avg_logits:
            return y.mean(dim=1)
        else:
            return F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)).float().mean(dim=1)


class CertifiedMuSTextClassifier(nn.Module):
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        quant: int = 64,
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.q = quant
        self.lambda_ = int(lambda_ * self.q) / self.q

        # Assume that the classifier comes with these
        self.embed_fn = base_classifier.get_input_embeddings()


    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ):
        # Exactly one of input_ids or inputs_embeds is used
        assert (input_ids is None) ^ (inputs_embeds is None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_fn(input_ids)

        bsz, L, _ = inputs_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs_embeds.device).long()

        if alpha is None:
            alpha = torch.ones_like(attention_mask)

        q = self.q
        all_ys = ()
        for x_, m_, a_ in zip(inputs_embeds, attention_mask, alpha):
            mus_masks = discretized_mus_masks(L, self.lambda_, q, device=x_.device) # (q, L)
            a_masked = a_.view(1, L) * mus_masks.view(q, L) # (q, L)
            
            # Mask the attention mask
            m_masked = m_.view(1, L) * a_masked # (q, L)
            x_big = x_.view(1, L, -1).repeat(q, 1, 1)

            y = self.base_classifier(inputs_embeds=x_big, attention_mask=m_masked)

            # Extract logits as necessary
            if hasattr(y, "logits"):
                y = y.logits
            elif isinstance(y, dict) and "logits" in y.keys():
                y = y["logits"]

            # Convert to one-hot and vote
            y = F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)) # (q, num_classes)
            avg_y = y.float().mean(dim=0) # (num_classes,)
            all_ys += (avg_y,)

        all_ys = torch.stack(all_ys) # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * self.lambda_)

        return {
            "logits": all_ys,   # (bsz, num_classes)
            "cert_rs": cert_rs  # (bsz,)
        }


class ChestXDataset(torch.utils.data.Dataset):
    """
    The ChestX-ray dataset adapted from torchxrayvision's curation of the GoogleNIH dataset.
    The structure segmentations are also derived using torchxrayvision's off-the-shelf models.

    This dataset is hosted on HuggingFace, see:
    https://huggingface.co/datasets/BrachioLab/chestx
    """

    pathology_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]

    structure_names: str = [
        "Left Clavicle",
        "Right Clavicle",
        "Left Scapula",
        "Right Scapula",
        "Left Lung",
        "Right Lung",
        "Left Hilus Pulmonis",
        "Right Hilus Pulmonis",
        "Heart",
        "Aorta",
        "Facies Diaphragmatica",
        "Mediastinum",
        "Weasand",
        "Spine"
    ]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = "BrachioLab/chestx",
        image_size: int = 224,
    ):
        r"""
        Args:
            split: Either "train" or "test"
            hf_data_repo: Where the dataset is hosted on HuggingFace
            image_size: The square image size
        """
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float().mean(dim=0, keepdim=True) / 255),
            tfs.Resize(image_size)

        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        image = self.preprocess_image(image)
        pathols = torch.tensor(self.dataset[idx]["pathols"])
        structs = torch.tensor(self.dataset[idx]["structs"])

        return {
            "image": image,     # (1,H,W)
            "pathols": pathols, # (14)
            "structs": structs, # (14,H,W)
        }


class ChestXPathologyModel(nn.Module, hfhub.PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.xrv_model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8

    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: (N,C,224,224) with values in [0,1], with either C=1 or C=2 channels
        """

        x = x * 2048 - 1024 # The xrv model requires some brazingo scaling
        out = self.xrv_model(x)
        """
        The torchxrayvision model outputs 18 pathology labels in the following order:
            [
                'Atelectasis',
                 'Consolidation',
                 'Infiltration',
                 'Pneumothorax',
                 'Edema',
                 'Emphysema',
                 'Fibrosis',
                 'Effusion',
                 'Pneumonia',
                 'Pleural_Thickening',
                 'Cardiomegaly',
                 'Nodule',
                 'Mass',
                 'Hernia',
                 '',
                 '',
                 '',
                 ''
            ]
        ... so we need to sort it to match our ordering
        """
        pathol_idxs = [0, 10, 1, 4, 7, 5, 6, 13, 2, 12, 11, 9, 8, 3]
        return out[:,pathol_idxs]


""" Fuck """
import sys
sys.path.append("/home/antonxue/foo/exlib/src")
import exlib
from exlib.explainers import ShapImageCls, LimeImageCls


def get_shap_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    num_patches: int = 196,
    num_samples: int = 1000,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)

    # call SHAP
    sek = {'max_evals': num_samples}
    explainer = ShapImageCls(model, shap_explainer_kwargs=sek)

    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(
        expln.attributions,
        image_size = image.shape[1],
        num_patches_one_axis = num_patches_one_axis,
        masked = True,
        top_k_frac = top_k_frac
    )
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha


def patch_segmenter(image, sz=(7,7), return_pt=False): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    x = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=x.size()[:2], mode='nearest').long()
    segments = segments[0,0]
    if not return_pt:
        segments = segments.numpy()
    return segments


def get_lime_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    patch_segmenter = patch_segmenter,
    num_patches: int = 196,
    num_samples: int = 1000,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)

    # call LIME
    eik = {
        "segmentation_fn": patch_segmenter,
        "top_labels": 1000, 
        "hide_color": 0, 
        "num_samples": num_samples
    }
    gimk = {
        "positive_only": False
    }
    liek = {
        "random_state": 1
    }

    explainer = LimeImageCls(model, 
                             explain_instance_kwargs=eik, 
                             get_image_and_mask_kwargs=gimk, 
                             LimeImageExplainerKwargs=liek)
    
    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(expln.attributions, image_size=image.shape[1], num_patches_one_axis=num_patches_one_axis, masked=True, top_k_frac=top_k_frac)
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha


def get_alpha_from_img_attrs(attrs, image_size=224, num_patches_one_axis=14, masked=True, top_k_frac=0.25):
    """
    Convert input tensor into a patch version by averaging over patches.
    
    Parameters:
        attrs (torch.Tensor): Input tensor of shape (N, C, image_size, image_size).
        image_size (int): Size of the input image (assumed square).
        num_patches_one_axis (int): Number of patches along one dimension.
        masked (bool): If True, return a boolean mask based on top_k_frac.
        top_k_frac (float): Fraction of top elements to include in the mask.
        
    Returns:
        torch.Tensor: Patch version of the input tensor of shape (N, num_patches_one_axis, num_patches_one_axis).
        (If masked=True, return the boolean mask where the threshold is top_k_frac).
    """
    # Calculate the approximate patch size and the leftover regions
    patch_size = image_size // num_patches_one_axis
    remainder = image_size % num_patches_one_axis

    # Generate dynamic patch sizes with adjusted padding for the last patch
    patch_sizes = [patch_size + 1 if i < remainder else patch_size for i in range(num_patches_one_axis)]

    # Cumulative sum to identify patch boundaries
    boundaries = [0] + torch.cumsum(torch.tensor(patch_sizes), 0).tolist()

    N, C, H, W = attrs.shape
    if H != image_size or W != image_size:
        raise ValueError(f"Input tensor height and width must match the given image_size ({image_size}).")
    
    # Aggregate values into patches
    patch_tensor = torch.zeros((N, num_patches_one_axis, num_patches_one_axis), device=attrs.device)
    for i in range(num_patches_one_axis):
        for j in range(num_patches_one_axis):
            patch = attrs[
                :, :, boundaries[i]:boundaries[i + 1], boundaries[j]:boundaries[j + 1]
            ]
            patch_tensor[:, i, j] = patch.mean(dim=(-1, -2, -3))

    result = patch_tensor
    if masked:
        threshold_num = int(top_k_frac * len(result[0].flatten()))
        result_masked = []
        for i in range(N):
            topk_indices = torch.topk(result[i].flatten(), threshold_num).indices
            result_masked_i = torch.zeros_like(result[i].flatten())
            result_masked_i[topk_indices] = 1
            result_masked.append(result_masked_i)
        result_masked = torch.stack(result_masked)
        return result_masked.view(result.shape).to(attrs.device)
    else:
        return result

