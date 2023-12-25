from torch import nn
import time
import torch


class CrossAttentionProfiler:
    def __init__(
            self,
            memory_len: int,
            num_query: int,
            bs: int,
            embed_dim: int,
            attn: nn.Module,
            output_path: str = None,
            warnup_times: int = 5,
            use_cuda: bool = False,
            profile_memory: bool = False,
            autograd: bool = True,
    ):
        device = torch.device('cuda')
        self.memory = torch.randn(memory_len, bs, embed_dim).to(device)
        self.pos_embed = torch.randn(memory_len, bs, embed_dim).to(device)
        self.query_embed = torch.randn(num_query, bs, embed_dim).to(device)
        self.target = torch.zeros_like(self.query_embed).to(device)
        self.mask = torch.zeros(3300, 16, 8, 32).to(device)
        # self.mask = torch.zeros(bs, memory_len, dtype=torch.bool).to(device)
        self.attn = attn.to(device)
        self.warnup_times = warnup_times
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory

        self.autograd = autograd
        self.output_path = output_path

    def __call__(self, *args, **kwargs):
        # Warn-up
        for _ in range(self.warnup_times):
            start = time.time()
            outputs = self.attn(query=self.target, key=self.memory, value=self.memory, query_pos=self.query_embed,
                                key_pos=self.pos_embed,
                                key_padding_mask=self.mask)
            torch.cuda.synchronize()
            end = time.time()
            print('Time:{}ms'.format((end - start) * 1000))

        if self.autograd:
            with torch.autograd.profiler.profile(enabled=True, use_cuda=self.use_cuda, record_shapes=False,
                                                 profile_memory=self.profile_memory) as prof:
                outputs = self.attn(query=self.target, key=self.memory, value=self.memory, query_pos=self.query_embed,
                                    key_pos=self.pos_embed,
                                    key_padding_mask=self.mask)
        else:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
                    record_shapes=True,
                    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                    with_stack=True
            ) as prof:
                outputs = self.attn(query=self.target, key=self.memory, value=self.memory, query_pos=self.query_embed,
                                    key_pos=self.pos_embed,
                                    key_padding_mask=self.mask)
        print(prof.table())
        if self.output_path is not None:
            prof.export_chrome_trace(self.output_path)


class SelfAttentionProfiler:
    def __init__(
            self,
            feature_len: int,
            bs: int,
            embed_dim: int,
            attn: nn.Module,
            batch_first: bool = False,
            output_path: str = None,
            warnup_times: int = 5,
            use_cuda: bool = False,
            profile_memory: bool = False,
            autograd: bool = False,
    ):
        device = torch.device('cuda')
        if not batch_first:
            self.input = torch.randn(feature_len, bs, embed_dim).to(device)
            self.query_pos = torch.randn(feature_len, bs, embed_dim).to(device)
        else:
            self.input = torch.randn(bs, feature_len, embed_dim).to(device)
            self.query_pos = torch.randn(bs, feature_len, embed_dim).to(device)

        self.key_padding_mask = torch.zeros(bs, feature_len, dtype=torch.bool).to(device)
        self.attn = attn.to(device)
        self.warnup_times = warnup_times
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory

        self.autograd = autograd

        self.output_path = output_path

    def __call__(self, *args, **kwargs):
        # Warn-up
        for _ in range(self.warnup_times):
            start = time.time()
            outputs = self.attn(query=self.input, key=self.input, value=self.input, query_pos=self.query_pos,
                                key_pos=self.query_pos,
                                key_padding_mask=self.key_padding_mask)
            torch.cuda.synchronize()
            end = time.time()
            print('Time:{}ms'.format((end - start) * 1000))

        if self.autograd:
            with torch.autograd.profiler.profile(enabled=True, use_cuda=self.use_cuda, record_shapes=False,
                                                 profile_memory=self.profile_memory) as prof:
                outputs = self.attn(query=self.input, key=self.input, value=self.input, query_pos=self.query_pos,
                                    key_pos=self.query_pos,
                                    key_padding_mask=self.key_padding_mask)
        else:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
                    record_shapes=True,
                    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                    with_stack=True
            ) as prof:
                outputs = self.attn(query=self.input, key=self.input, value=self.input, query_pos=self.query_pos,
                                    key_pos=self.query_pos,
                                    key_padding_mask=self.key_padding_mask)
        print(prof.table())
        if self.output_path is not None:
            prof.export_chrome_trace(self.output_path)
