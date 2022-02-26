# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)

from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put


logger = logging.getLogger(__name__)


@dataclass
class Data2VecAudioConfig(Wav2Vec2Config):

    # l1 smooth loss의 하이퍼 파라메터
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    # 상위 k 개 레이어의 output을 averaging해서 타겟으로 사용함. 
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    # EMA decay 하이퍼 파라메터
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_audio", dataclass=Data2VecAudioConfig)
class Data2VecAudioModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioConfig):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers) 
        '''
        (Pdb) cfg.conv_feature_layers
        '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
        (Pdb) eval
        <built-in function eval>

        (Pdb) feature_enc_layers
        [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
        '''

        self.extractor_embed = feature_enc_layers[-1][0] # 512

        self.ema = None
        self.embed = cfg.encoder_embed_dim # 768

        # top-k layer
        self.average_top_k_layers = cfg.average_top_k_layers # 8

        # loss
        self.loss_beta = cfg.loss_beta # 0.0
        self.loss_scale = cfg.loss_scale # None

        # 7 layer 1d CNN
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers, # [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
            dropout=0.0,
            mode=cfg.extractor_mode, # layer_norm
            conv_bias=cfg.conv_bias, # False
        )

        '''
        (Pdb) self.feature_extractor
        ConvFeatureExtractionModel(
        (conv_layers): ModuleList(
            (0): Sequential(
            (0): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
            (1): Dropout(p=0.0, inplace=False)
            (2): Sequential(
                (0): TransposeLast()
                (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (2): TransposeLast()
            )
            ...

            (3): GELU()
            )
            (6): Sequential(
            (0): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
            (1): Dropout(p=0.0, inplace=False)
            (2): Sequential(
                (0): TransposeLast()
                (1): Fp32LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (2): TransposeLast()
            )
            (3): GELU()
            )
        )
        )
        '''

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim) 
        # Linear(in_features=512, out_features=768, bias=True)

        self.mask_prob = cfg.mask_prob # 0.65
        self.mask_selection = cfg.mask_selection # static
        self.mask_other = cfg.mask_other # 0.0
        self.mask_length = cfg.mask_length # 10
        self.no_mask_overlap = cfg.no_mask_overlap # False
        self.mask_min_space = cfg.mask_min_space # 1

        self.mask_channel_prob = cfg.mask_channel_prob # 0.0
        self.mask_channel_before = cfg.mask_channel_before # False
        self.mask_channel_selection = cfg.mask_channel_selection # static
        self.mask_channel_other = cfg.mask_channel_other # 0.0
        self.mask_channel_length = cfg.mask_channel_length # 10
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap # False
        self.mask_channel_min_space = cfg.mask_channel_min_space # 1

        self.dropout_input = nn.Dropout(cfg.dropout_input) 
        # Dropout(p=0.0, inplace=False), maybe for raw audio

        self.dropout_features = nn.Dropout(cfg.dropout_features) 
        # Dropout(p=0.0, inplace=False), maybe for features after 7 layer 1d cnn

        self.feature_grad_mult = cfg.feature_grad_mult # 1.0

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        ) # (Pdb) self.mask_emb.size(), torch.Size([768])

        # import pdb; pdb.set_trace()

        self.encoder = TransformerEncoder(cfg)
        # transformer 뿐만 아니라 pos conv 가 있음.
        '''
        TransformerEncoder(
        (pos_conv): Sequential(
            (0): Sequential(
            (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
            (1): SamePad()
            (2): TransposeLast()
            (3): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=False)
            (4): TransposeLast()
            (5): GELU()
            )
            ...
            (4): Sequential(
            (0): Conv1d(768, 768, kernel_size=(19,), stride=(1,), padding=(9,), groups=16)
            (1): SamePad()
            (2): TransposeLast()
            (3): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=False)
            (4): TransposeLast()
            (5): GELU()
            )
        )
        (layers): ModuleList(
            (0): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
                (dropout_module): FairseqDropout()
                (k_proj): Linear(in_features=768, out_features=768, bias=True)
                (v_proj): Linear(in_features=768, out_features=768, bias=True)
                (q_proj): Linear(in_features=768, out_features=768, bias=True)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
            )
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.0, inplace=False)
            (dropout3): Dropout(p=0.1, inplace=False)
            (self_attn_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (final_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
            )
            ...
            (11): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
                (dropout_module): FairseqDropout()
                (k_proj): Linear(in_features=768, out_features=768, bias=True)
                (v_proj): Linear(in_features=768, out_features=768, bias=True)
                (q_proj): Linear(in_features=768, out_features=768, bias=True)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
            )
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.0, inplace=False)
            (dropout3): Dropout(p=0.1, inplace=False)
            (self_attn_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (final_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
            )
        )
        (layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
        )
        '''

        self.layer_norm = LayerNorm(self.extractor_embed) 
        # FusedLayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)

        self.final_proj = nn.Linear(self.embed, self.embed) 
        # Linear(in_features=768, out_features=768, bias=True)

        self.num_updates = 0


    # Exponential Moving Average teacher 를 만들어야함.
    # ∆ ← τ ∆ + (1 − τ ) θ
    # \delta <- \tau * \delta + (1-\tau) * \theta
    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay, # 0.999 default
            ema_fp32=True,
        )
        # 전체 모델이 7 layer 1d-cnn + 5 layer pos conv + 12 layer transformer encoder 인데
        # pos conv를 제외하나?
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        # student model을 deepcopy 하는 모듈임
        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

        '''
        args : model, config, device, skip_keys
        return : copied model
        '''

    
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        # 최초에 EMA teacher를 만듬
        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            # ema.set_decay(), ema.step() 이 필요함.
            # ∆ ← τ ∆ + (1 − τ ) θ
            # \delta <- \tau * \delta + (1-\tau) * \theta
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates


    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state


    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


    # build model
    @classmethod
    def build_model(cls, cfg: Data2VecAudioConfig, task=None):
        """Build a new model instance."""
        
        return cls(cfg)


    # apply mask after feature extract (1d cnn in w2v2)
    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
        mask_prob=None
    ):
        B, T, C = x.shape
        if mask_prob is not None:
            current_mask_prob = mask_prob
        else:
            current_mask_prob = self.mask_prob
            

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if current_mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    current_mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices


    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)


    def forward(
        self,
        source, # wav input
        padding_mask=None, # ??
        mask=True, 
        features_only=False, # SSL 할거냐
        layer=None, 
        mask_indices=None, # input mask
        mask_channel_indices=None,
        padding_count=None,
        cnn_features_only=False,
    ):
        features = source

        # 1d cnn 으로 down sampling 한 feautre를 추출함.
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
                # GradMultiply는 아마 단순히 gradient의 magnitude를 건드리는듯?
                '''
                class GradMultiply(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x, scale):
                        ctx.scale = scale
                        res = x.new(x)
                        return res

                    @staticmethod
                    def backward(ctx, grad):
                        return grad * ctx.scale, None
                '''
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        # import pdb; pdb.set_trace()

        # 1d conv 이후의 output_lengths 를 계산하고 이를 바탕으로 padding mask를 다시 만드는?
        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None


        # 1d cnn을 통과시킨 뒤에 post projection을 또 하는 이유는 뭘까? 
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)


        # w2v2가 1d cnn + pos conv + transformer인데
        # ema를 transformer부분만 할건지, 1d cnn + transformer 둘 다 할건지를 정함?
        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        # self.dropout_features 는 그럼 어디에 쓰이지?
        features = self.dropout_input(features)

        # import pdb; pdb.set_trace()

        # 1d cnn을 통과한 (down sampling된) feature를 마스킹함.
        # finetuning을 할 때는 masking이 필요없어서 if, else로 나눈듯
        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
                mask_prob=None
            )
        else:
            x = features
            mask_indices = None

        # import pdb; pdb.set_trace()
        '''
        (Pdb) mask_indices.size()
        torch.Size([8, 634])
        (Pdb) features.size()
        torch.Size([8, 634, 768])
        '''


        if cnn_features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
            }

        # pos conv + transformer 를 통과시킴
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        # finetuning 시에 사용함, finetuning 시에는 ema representation이 필요 없음.
        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }


        ## ema training 할 경우
        # result bucket
        result = {
            "losses": {},
        }

        # teacher mode의 representation을 뽑으려면 이는 업데이트되는게 아니므로
        # no grad, eval mode 가 필요함.
        with torch.no_grad():
            self.ema.model.eval()

            # whether to momentum update only the transformer
            # 즉 pos conv 말고 transformer만 업데이트 할건지를 정하는듯?
            if self.cfg.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features, # pre_encoder_features 가 None 이냐 featrues.clone() 이냐
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers, # top-k 개수만 제외하고
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y = self.ema.model.extract_features(
                    source=source,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]


        # x는 여전히 wav2vec encoder output
        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        # teacher - student 간 l1 loss 계산 (smooth or not)
        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)


        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)


        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()


        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())


        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )


        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result


    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()


    # finetuning에 사용됨 features_only=True
    def extract_features(
        self, source, padding_mask, mask=False, layer=None, cnn_features_only=False,
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
            cnn_features_only=cnn_features_only,
        )
        return res


    # 나중에 finetuning 할 때 pretraining에 사용한 ema와 projection layer 제거
    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
