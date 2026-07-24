from pathlib import Path

import torch

from prismatic.models.pair_bridge import (
    PairBridge,
    PairBridgeConfig,
    alignment_loss,
    build_pair_perception_tokens,
    cosine_alignment_loss,
    load_pair_bridge_checkpoint,
    save_pair_bridge_checkpoint,
)
from prismatic.vla.constants import IGNORE_INDEX


def test_pair_bridge_shapes_and_gate_init(tmp_path: Path):
    config = PairBridgeConfig(llm_dim=4096, bridge_dim=512, latent_dim=16, horizon=8, action_dim=7)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 20, 4096)
    base_init = torch.randn(2, 8, 4096)

    output = bridge(perception_tokens, base_init)

    assert output.action_init.shape == (2, 8, 4096)
    assert output.action_init_delta.shape == (2, 8, 4096)
    assert output.z_align.shape == (2, 8, 16)
    assert output.init_gate.shape == (2, 8)
    assert output.init_gate_raw.shape == (2, 8)
    assert torch.allclose(output.init_gate, torch.full((2, 8), config.init_gate_value))
    assert not torch.allclose(output.action_init, base_init)
    assert bridge.config.init_gate_granularity == "per_step"
    assert bridge.config.input_dependent_gate
    assert "init_gate" not in dict(bridge.named_parameters())
    assert dict(bridge.named_parameters())["align_proj.1.weight"].shape == (512, 512)
    assert dict(bridge.named_parameters())["align_proj.3.weight"].shape == (16, 512)
    assert dict(bridge.named_parameters())["init_proj.1.weight"].shape == (2048, 512)
    assert dict(bridge.named_parameters())["init_proj.3.weight"].shape == (4096, 2048)
    assert dict(bridge.named_parameters())["gate_proj.weight"].shape == (1, 512)
    assert "slot_scale" not in dict(bridge.named_parameters())
    assert torch.count_nonzero(bridge.gate_proj.weight) == 0
    expected_bias = torch.logit(torch.tensor(config.init_gate_value))
    assert torch.allclose(bridge.gate_proj.bias, expected_bias.reshape(1))
    assert bridge.config.bridge_mlp_dim == 2048
    assert bridge.config.init_mlp_dim == 2048
    assert bridge.config.gate_num_layers == 1
    assert bridge.config.gate_mlp_dim == 256
    assert bridge.cross_block.cross_attn.embed_dim == 512
    assert bridge.self_block.self_attn.embed_dim == 512
    assert dict(bridge.named_parameters())["cross_block.mlp.0.weight"].shape == (2048, 512)
    assert dict(bridge.named_parameters())["self_block.mlp.3.weight"].shape == (512, 2048)

    ckpt = tmp_path / "pair_bridge.pt"
    save_pair_bridge_checkpoint(
        path=ckpt,
        pair_bridge=bridge,
        config=config,
        action_ae_encoder_path="/tmp/encoder.pt",
        metadata={"step": 0},
    )
    loaded = load_pair_bridge_checkpoint(ckpt)
    loaded_output = loaded(perception_tokens, base_init)

    assert loaded_output.action_init.shape == (2, 8, 4096)
    assert loaded_output.action_init_delta.shape == (2, 8, 4096)
    assert loaded_output.z_align.shape == (2, 8, 16)
    assert loaded_output.init_gate.shape == (2, 8)


def test_pair_bridge_two_layer_gate_config():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        gate_num_layers=2,
        gate_mlp_dim=16,
    )
    bridge = PairBridge(config)

    assert dict(bridge.named_parameters())["gate_proj.0.weight"].shape == (16, 32)
    assert dict(bridge.named_parameters())["gate_proj.2.weight"].shape == (1, 16)
    assert torch.count_nonzero(bridge.gate_proj[2].weight) == 0


def test_pair_perception_helper_training_and_inference_masks():
    hidden_state = torch.randn(2, 10, 4)
    labels = torch.full((2, 7), IGNORE_INDEX)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
        ]
    )

    train_tokens, train_mask = build_pair_perception_tokens(
        hidden_state=hidden_state,
        labels=labels,
        attention_mask=attention_mask,
        num_patches=3,
    )

    assert train_tokens.shape == (2, 9, 4)
    assert torch.equal(train_mask[:, :3], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(train_mask[0, 3:], torch.tensor([True, True, True, True, False, False]))
    assert torch.equal(train_mask[1, 3:], torch.tensor([True, True, True, False, False, False]))

    infer_tokens, infer_mask = build_pair_perception_tokens(
        hidden_state=hidden_state[:1],
        attention_mask=torch.tensor([[1, 1, 0, 1, 1, 1, 1]]),
        num_patches=3,
        num_prompt_tokens=4,
    )

    assert infer_tokens.shape == (1, 7, 4)
    assert torch.equal(infer_mask[0], torch.tensor([True, True, True, True, True, False, True]))


def test_pair_bridge_dual_heads_receive_gradients():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 8, 64)

    output = bridge(perception_tokens, base_init)
    loss = output.z_align.float().sum() + output.action_init_delta.float().sum()
    loss.backward()

    assert bridge.align_proj[1].weight.grad is not None
    assert bridge.align_proj[1].weight.grad.abs().sum() > 0
    assert bridge.align_proj[3].weight.grad is not None
    assert bridge.align_proj[3].weight.grad.abs().sum() > 0
    assert bridge.init_proj[1].weight.grad is not None
    assert bridge.init_proj[1].weight.grad.abs().sum() > 0
    assert bridge.init_proj[3].weight.grad is not None
    assert bridge.init_proj[3].weight.grad.abs().sum() > 0


def test_pair_bridge_fixed_gate_mode():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_mode="fixed",
        init_gate_value=0.1,
        init_gate_granularity="scalar",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 8, 64)

    output = bridge(perception_tokens, base_init)

    assert torch.allclose(output.init_gate, torch.tensor(0.1))
    assert "init_gate" not in dict(bridge.named_parameters())
    assert "init_gate" in dict(bridge.named_buffers())


def test_pair_bridge_fixed_gate_mode_accepts_exact_one_with_tanh():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_mode="fixed",
        init_gate_value=1.0,
        init_gate_granularity="per_step",
        gate_activation="tanh",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.randn(2, 8, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (8,)
    assert torch.allclose(output.init_gate, torch.ones(8))
    assert torch.allclose(
        output.action_init,
        base_init + output.action_init_delta,
    )


def test_pair_bridge_keeps_gate_fp32_after_bf16_cast():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config).to(torch.bfloat16)

    bridge.keep_high_precision_params()

    assert bridge.down_proj.weight.dtype == torch.bfloat16
    assert bridge.cross_block.cross_attn.in_proj_weight.dtype == torch.bfloat16
    assert bridge.self_block.self_attn.in_proj_weight.dtype == torch.bfloat16
    assert bridge.gate_norm.weight.dtype == torch.float32
    assert bridge.gate_proj.weight.dtype == torch.float32
    assert dict(bridge.named_parameters())["gate_proj.weight"].dtype == torch.float32


def test_pair_bridge_bf16_forward_with_fp32_gate_on_cuda():
    if not torch.cuda.is_available():
        return

    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config).to(torch.bfloat16).cuda()
    bridge.keep_high_precision_params()

    perception_tokens = torch.randn(2, 6, 64, device="cuda", dtype=torch.bfloat16)
    base_init = torch.zeros(2, 8, 64, device="cuda", dtype=torch.bfloat16)
    perception_mask = torch.ones(2, 6, device="cuda", dtype=torch.bool)

    output = bridge(perception_tokens, base_init, perception_mask)

    assert output.action_init.shape == (2, 8, 64)
    assert output.init_gate.shape == (2, 8)
    assert output.action_init.dtype == torch.bfloat16
    assert output.init_gate.dtype == torch.bfloat16


def test_pair_bridge_per_step_gate_broadcasts_across_steps():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_value=0.2,
        init_gate_granularity="per_step",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.randn(2, 8, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (2, 8)
    expected = base_init + output.init_gate.view(2, 8, 1) * output.action_init_delta
    assert torch.allclose(output.action_init, expected)


def test_pair_bridge_input_dependent_gate_changes_with_tokens():
    torch.manual_seed(23)
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    with torch.no_grad():
        bridge.gate_proj.weight[:, 0] = 0.25
        bridge.gate_proj.bias.zero_()
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 8, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (2, 8)
    assert output.init_gate.std(unbiased=False) > 0


def test_pair_bridge_perception_mask():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 8, 64)
    perception_mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, False]])

    output = bridge(perception_tokens, base_init, perception_mask)

    assert output.action_init.shape == (2, 8, 64)
    assert output.z_align.shape == (2, 8, 8)


def test_pair_bridge_respects_configured_latent_dim():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 8, 64)

    output = bridge(perception_tokens, base_init)

    assert output.z_align.shape == (2, 8, 8)


def test_pair_alignment_loss_supports_cosine_and_l2():
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [1.0, 2.0]]])

    cosine = alignment_loss(predicted, target, loss_type="cosine")
    l2 = alignment_loss(predicted, target, loss_type="l2")

    assert torch.allclose(cosine, cosine_alignment_loss(predicted, target))
    assert torch.allclose(l2, torch.tensor(3.25))

    l2.backward()
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0


def test_pair_alignment_loss_rejects_invalid_type_and_shape():
    predicted = torch.zeros(1, 8, 16)
    target = torch.zeros(1, 8, 16)

    try:
        alignment_loss(predicted, target, loss_type="l1")
    except ValueError as exc:
        assert "expected 'cosine' or 'l2'" in str(exc)
    else:
        raise AssertionError("Expected invalid alignment loss type to raise ValueError")

    try:
        alignment_loss(predicted, target[:, :, :8], loss_type="l2")
    except ValueError as exc:
        assert "must have same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched alignment tensor shapes to raise ValueError")
