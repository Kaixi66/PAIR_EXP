import os
import subprocess
import sys

import tensorflow as tf

from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs


def test_uf850_platform_constants_are_explicit():
    env = dict(os.environ, VLA_ROBOT_PLATFORM="UF850")
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "from prismatic.vla.constants import *; "
                "from prismatic.models.projectors import ProprioProjector; "
                "p=ProprioProjector(llm_dim=16, proprio_dim=PROPRIO_DIM); "
                "print(ROBOT_PLATFORM, NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM, p.fc1.in_features)"
            ),
        ],
        env=env,
        text=True,
    )
    assert "UF850 8 7 6 6" in output


def test_uf850_oxe_alias_preserves_true_six_joint_semantics(tmp_path):
    kwargs = make_oxe_dataset_kwargs(
        "uf850_vr_teleop_rlds",
        tmp_path,
        load_camera_views=("primary", "wrist"),
        load_proprio=True,
        load_language=True,
    )
    assert kwargs["name"] == "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
    assert kwargs["state_obs_keys"] == ["joint_angles"]
    assert kwargs["absolute_action_mask"] == [False] * 6 + [True]
    assert kwargs["action_normalization_mask"] == [True] * 6 + [False]

    joint_angles = tf.zeros((3, 6), dtype=tf.float32)
    trajectory = {"observation": {"end_effector_pose": joint_angles}}
    transformed = kwargs["standardize_fn"](trajectory)
    assert transformed["observation"]["joint_angles"].shape == (3, 6)
    assert transformed["observation"]["joint_angles"] is joint_angles


def test_action_chunk_tail_uses_neutral_relative_and_held_absolute_actions():
    actions = tf.constant(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, -1.0],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 1.0],
            [7.0, 8.0, 9.0, 0.7, 0.8, 0.9, 1.0],
        ],
        dtype=tf.float32,
    )
    trajectory = {
        "action": actions,
        "observation": {"proprio": tf.zeros((3, 6), dtype=tf.float32)},
        "task": {"language_instruction": tf.constant(["task"] * 3)},
        "dataset_name": tf.constant(["uf850"] * 3),
        "absolute_action_mask": tf.constant([[False] * 6 + [True]] * 3),
    }

    chunked = chunk_act_obs(trajectory, window_size=1, future_action_window_size=2)
    last_chunk = chunked["action"][-1]

    assert chunked["action"].shape == (3, 3, 7)
    tf.debugging.assert_equal(last_chunk[0], actions[-1])
    tf.debugging.assert_equal(last_chunk[1:, :6], tf.zeros((2, 6)))
    tf.debugging.assert_equal(last_chunk[:, 6], tf.ones(3))
