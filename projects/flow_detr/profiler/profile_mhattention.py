from detrex.layers.attention import MultiheadAttention
from utils.attn_profiler import CrossAttentionProfiler, SelfAttentionProfiler

if __name__ == '__main__':
    attn = MultiheadAttention(256, 8)
    # attn_type = "encoder_self"
    # attn_type = "decoder_self"
    attn_type = "cross"
    if attn_type == "cross":
        attn_profiler = CrossAttentionProfiler(memory_len=3400, num_query=100, bs=16, embed_dim=256,
                                               attn=attn,
                                               use_cuda=True,
                                               profile_memory=False, autograd=True
                                               # output_path="/root/autodl-tmp/profiler_output/flow_attention_profiler_2.json"
                                               )
    elif attn_type == "decoder_self":
        attn_profiler = SelfAttentionProfiler(feature_len=100, bs=16, embed_dim=256, attn=attn, use_cuda=True,
                                              profile_memory=False, autograd=True
                                              # output_path="/root/autodl-tmp/profiler_output/decoder_self_attention_profiler.json"
                                              )
    else:
        attn_profiler = SelfAttentionProfiler(feature_len=13600, bs=16, embed_dim=256, attn=attn, use_cuda=True,
                                              profile_memory=False, autograd=True
                                              # output_path="/root/autodl-tmp/profiler_output/flow_self_attention_profiler.json"
                                              )
    attn_profiler()
