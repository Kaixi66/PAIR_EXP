import torch
from torch import nn

from prismatic.models.action_heads import L1RegressionActionHead, MLPResNet
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


def test_action_head_initial_action_states_preserve_zero_default():
    torch.manual_seed(7)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25

    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    proprio_projector = nn.Linear(8, hidden_dim).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    zero_init = torch.zeros(batch_size, 7 * NUM_ACTIONS_CHUNK, hidden_dim)

    default_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
    )
    explicit_zero_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=zero_init,
    )

    assert default_actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
    assert torch.allclose(default_actions, explicit_zero_actions)


def test_action_head_predict_action_without_proprio_forward():
    torch.manual_seed(11)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)

    actions = action_head.predict_action(
        hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Inference",
    )

    assert actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)


def test_action_head_post_stem_pair_gate_controls_injection():
    torch.manual_seed(13)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    proprio_projector = nn.Linear(8, hidden_dim).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    pair_init = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)

    default_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
    )
    zero_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.tensor(0.0),
    )
    nonzero_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.tensor(0.5),
    )

    assert torch.allclose(default_actions, zero_gate_actions)
    assert not torch.allclose(default_actions, nonzero_gate_actions)


def test_action_head_accepts_per_step_pair_gate():
    torch.manual_seed(17)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    proprio_projector = nn.Linear(8, hidden_dim).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    pair_init = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)

    zero_step_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.zeros(NUM_ACTIONS_CHUNK),
    )
    nonzero_step_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.linspace(0.1, 0.8, NUM_ACTIONS_CHUNK),
    )

    assert zero_step_gate_actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
    assert not torch.allclose(zero_step_gate_actions, nonzero_step_gate_actions)


def test_action_head_accepts_batched_per_step_pair_gate():
    torch.manual_seed(19)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    proprio_projector = nn.Linear(8, hidden_dim).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    pair_init = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)

    zero_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.zeros(batch_size, NUM_ACTIONS_CHUNK),
    )
    nonzero_gate_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=pair_init,
        initial_action_gate=torch.stack(
            [
                torch.linspace(0.1, 0.8, NUM_ACTIONS_CHUNK),
                torch.linspace(0.8, 0.1, NUM_ACTIONS_CHUNK),
            ]
        ),
    )

    assert zero_gate_actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
    assert not torch.allclose(zero_gate_actions, nonzero_gate_actions)


class _DoubleBlock(nn.Module):
    def forward(self, x, h_t=None, h_a=None, p=None):
        return 2.0 * x


def _boundary_test_model():
    model = MLPResNet(
        num_blocks=24,
        input_dim=1,
        hidden_dim=1,
        output_dim=1,
    )
    model.layer_norm1 = nn.Identity()
    model.fc1 = nn.Identity()
    model.relu = nn.Identity()
    model.mlp_resnet_blocks = nn.ModuleList([_DoubleBlock() for _ in range(24)])
    model.layer_norm2 = nn.Identity()
    model.fc2 = nn.Identity()
    return model


def test_pair_injections_occur_at_exact_expert_boundaries():
    model = _boundary_test_model()
    x = torch.zeros(1, NUM_ACTIONS_CHUNK, 1)
    delta = torch.ones_like(x)
    expert_hidden = torch.zeros(1, 25, 1)

    start = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gates={"start": torch.tensor(1.0)},
    )
    middle = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gates={"middle": torch.tensor(1.0)},
    )
    end = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gates={"end": torch.tensor(1.0)},
    )
    all_positions = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gates={
            "start": torch.tensor(1.0),
            "middle": torch.tensor(1.0),
            "end": torch.tensor(1.0),
        },
    )

    assert torch.equal(start, torch.full_like(start, 2.0**24))
    assert torch.equal(middle, torch.full_like(middle, 2.0**12))
    assert torch.equal(end, torch.ones_like(end))
    assert torch.equal(all_positions, start + middle + end)


def test_multi_position_zero_gates_match_no_pair_and_legacy_start_matches_mapping():
    torch.manual_seed(31)
    hidden_dim = 16
    batch_size = 2
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=2,
        use_pro_version=False,
    )
    hidden_states = torch.randn(batch_size, 25, 2 + NUM_TOKENS, hidden_dim)
    pair_delta = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)

    baseline = action_head.predict_action(hidden_states, phase="Inference")
    zero_gate = action_head.predict_action(
        hidden_states,
        phase="Inference",
        initial_action_states=pair_delta,
        pair_injection_gates={
            "start": torch.tensor(0.0),
            "middle": torch.zeros(NUM_ACTIONS_CHUNK),
            "end": torch.zeros(batch_size, NUM_ACTIONS_CHUNK),
        },
    )
    legacy_start = action_head.predict_action(
        hidden_states,
        phase="Inference",
        initial_action_states=pair_delta,
        initial_action_gate=torch.linspace(0.1, 0.8, NUM_ACTIONS_CHUNK),
    )
    mapped_start = action_head.predict_action(
        hidden_states,
        phase="Inference",
        initial_action_states=pair_delta,
        pair_injection_gates={
            "start": torch.linspace(0.1, 0.8, NUM_ACTIONS_CHUNK),
        },
    )

    assert torch.allclose(baseline, zero_gate)
    assert torch.allclose(legacy_start, mapped_start)


def test_multi_position_gates_and_shared_delta_receive_gradients():
    model = _boundary_test_model()
    x = torch.zeros(1, NUM_ACTIONS_CHUNK, 1)
    delta = torch.ones_like(x, requires_grad=True)
    gates = {
        position: torch.tensor(0.5, requires_grad=True)
        for position in ("start", "middle", "end")
    }
    expert_hidden = torch.zeros(1, 25, 1)

    output = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gates=gates,
    )
    output.sum().backward()

    assert delta.grad is not None
    assert delta.grad.abs().sum() > 0
    assert all(gate.grad is not None and gate.grad.abs().sum() > 0 for gate in gates.values())


def test_action_head_multi_position_pro_blocks_preserve_shape():
    torch.manual_seed(37)
    hidden_dim = 16
    batch_size = 1
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=2,
        use_pro_version=True,
    )
    hidden_states = torch.randn(batch_size, 25, 2 + NUM_TOKENS, hidden_dim)
    pair_delta = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)

    actions = action_head.predict_action(
        hidden_states,
        phase="Inference",
        initial_action_states=pair_delta,
        pair_injection_gates={
            "start": torch.tensor(0.1),
            "middle": torch.full((NUM_ACTIONS_CHUNK,), 0.2),
            "end": torch.full((batch_size, NUM_ACTIONS_CHUNK), 0.3),
        },
    )

    assert actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
