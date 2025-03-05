
import logging

from model import LanguageModelConfig, DigitUltimateConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def main():
    DigitUltimate  = LanguageModelConfig(
        vocab_size=4096000*2,
        pad_token=0,
        eos_token=2,
        sequence_len: int = 131072*2,
        embedding_init_scale=0.05,
        dim_int = 138456,
        seq_len: int = 32768,     # 32K context window
        expert_capacity: float = 2.0,
        attn_type: str = "hybrid"  # [vanilla, hybrid, sparse]
        quant_mode: str = "fp8"    # [fp32, fp16, fp8, int8]
        mesh_shape = (4000, 2)  # مثال: 8 دستگاه (4 داده، 2 مدل)
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = Mesh(devices, axis_names=('data', 'model'))
        activation: str = "swiglu" # [gelu, swish, swiglu]
        precision: str = "bf16"   # Com
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=DigitUltimateConfig(
            emb_size=131072*2,
            widening_factor=64,
            key_size=2048,
            num_q_heads=2048,
            num_kv_heads=1024,
            num_layers=8192,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            enable_quantum_entanglement: bool = True,
            enable_hierarchical_search: bool = True,
            enable_quant= True
    
            # MoE.
            num_experts=512,
            num_selected_experts=24,
            # Activation sharding.
            flash_mesh=mesh,
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=DigitUltimate ,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=2000, temperature=0.01))
def __post_init__(self):
        assert self.dim % 128 == 0, "Dimension must be divisible by 128"
        assert self.num_heads % 8 == 0, "Heads must be divisible by 8"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
