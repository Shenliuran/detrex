from utils.flow_attention import LinearAttention, LinearCrossAttention
from utils.attn_profiler import CrossAttentionProfiler, SelfAttentionProfiler

if __name__ == '__main__':
    attn = LinearCrossAttention(256, 8, competition_temperature=0.1)
    # attn_type = "decoder_self"
    # attn_type = "cross"
    attn_type = "cross"
    if attn_type == "cross":
        flow_attn_profiler = CrossAttentionProfiler(memory_len=3400, num_query=100, bs=16, embed_dim=256,
                                                    attn=attn,
                                                    use_cuda=True,
                                                    profile_memory=False, autograd=True,
                                                    output_path="/root/autodl-tmp/profiler_output/flow_attention_profile_profiler_2.json"
                                                    )
    elif attn_type == "decoder_self":
        flow_attn_profiler = SelfAttentionProfiler(feature_len=100, bs=16, embed_dim=256, attn=attn, use_cuda=True,
                                                   profile_memory=False, autograd=True,
                                                   # output_path="/root/autodl-tmp/profiler_output/flow_prof_decoder_self_attention_profiler.json"
                                                   )
    else:
        flow_attn_profiler = SelfAttentionProfiler(feature_len=3250, bs=16, embed_dim=256, attn=attn, use_cuda=True,
                                                   profile_memory=False, autograd=True,
                                                   # output_path="/root/autodl-tmp/profiler_output/flow_prof_encoder_self_attention_profiler_1000.json"
                                                   )
    flow_attn_profiler()
