# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for staggered-agent-death handling in multi-agent environments.

Problem scenario:
  When agents die at different timesteps in a NEXT_STEP autoreset environment,
  RLlib stops sending actions for dead agents after their terminal step. The
  Python side (RayEnv/RayVecEnv) forwards exactly the actions provided by
  RLlib — no zero-padding is added for dead agents.

  On the C++ side, TScholaEnvironment::Step() handles missing-action agents:
    1. It snapshots agents that are already terminated/truncated.
    2. It filters dead agents out of InActions before calling Execute_Step(),
       so the Blueprint implementation only receives actions for live agents.
    3. It restores the dead agents' terminal flags after Execute_Step(), so
       AllAgentsCompleted() can eventually return true and end the episode.

Python-side contract tested here:
  - RayEnv.step() forwards the caller's action dict unchanged to the protocol.
  - __all__ is computed correctly as agents die one by one.
  - The episode is considered complete once every agent has been marked dead.

Run:
  python -m pytest Test/rllib/test_staggered_death.py -v
  OR (standalone, no pytest required):
  python Test/rllib/test_staggered_death.py
"""

import sys
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# FakeProtocol — simulates staggered 3-agent death, no Unreal required
# ---------------------------------------------------------------------------
class FakeProtocol:
    """
    Mock protocol that kills agents one per step.

    Timeline:
      step 1: agent_0 dies (terminated=True), agent_1/agent_2 alive
      step 2: agent_1 dies,                  agent_2 alive
      step 3: agent_2 dies  -> all agents dead -> __all__ can be True
    """

    def __init__(self):
        self.step_count = 0
        self.agent_ids = ["agent_0", "agent_1", "agent_2"]
        self.env_id = 0
        self._started = True
        self.received_actions_log = []  # list[set] — one entry per step

    def __bool__(self):
        return self._started

    def start(self):
        self._started = True

    def close(self):
        pass

    @property
    def properties(self):
        return {}

    def send_startup_msg(self, auto_reset_type=None):
        pass

    def get_definition(self):
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        ids = [self.agent_ids]
        agent_types = [{a: "default" for a in self.agent_ids}]
        obs_defns = {0: {a: obs_space for a in self.agent_ids}}
        act_defns = {0: {a: act_space for a in self.agent_ids}}
        return ids, agent_types, obs_defns, act_defns

    def send_reset_msg(self, seeds=None, options=None):
        self.step_count = 0
        self.received_actions_log = []
        obs = [{a: np.zeros(4, dtype=np.float32) for a in self.agent_ids}]
        infos = [{a: {} for a in self.agent_ids}]
        return obs, infos

    def send_action_msg(self, actions, action_spaces):
        self.step_count += 1
        env_actions = actions[self.env_id]
        # Record exactly what keys were received (C++ TScholaEnvironment would
        # see the same set when Python is not doing any no-op padding).
        self.received_actions_log.append(set(env_actions.keys()))

        obs = {a: np.random.randn(4).astype(np.float32) for a in self.agent_ids}
        rewards = {a: 0.0 for a in self.agent_ids}
        terminateds = {a: False for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        infos = {a: {} for a in self.agent_ids}

        if self.step_count >= 1:
            terminateds["agent_0"] = True
            rewards["agent_0"] = -1.0
        if self.step_count >= 2:
            terminateds["agent_1"] = True
            rewards["agent_1"] = -1.0
        if self.step_count >= 3:
            terminateds["agent_2"] = True
            rewards["agent_2"] = -1.0

        return (
            {self.env_id: obs},
            {self.env_id: rewards},
            {self.env_id: terminateds},
            {self.env_id: truncateds},
            {self.env_id: infos},
            {},  # initial_obs  (unused in NEXT_STEP mode)
            {},  # initial_infos (unused in NEXT_STEP mode)
        )


class FakeSimulator:
    supported_protocols = (type(None),)

    def start(self, properties=None):
        pass

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Helper: build a RayEnv around FakeProtocol without starting Unreal
# ---------------------------------------------------------------------------
def make_ray_env():
    """Construct a RayEnv backed by FakeProtocol (no Unreal connection)."""
    protocol = FakeProtocol()
    try:
        from schola.rllib.env import RayEnv
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        simulator = FakeSimulator()
        simulator.supported_protocols = type(protocol)

        env = RayEnv.__new__(RayEnv)
        env.protocol = protocol
        env.simulator = simulator
        env._init_space_attributes()
        protocol.start()
        env.protocol.send_startup_msg()
        env._define_environment()
        env._init_agent_tracking()
        MultiAgentEnv.__init__(env)
        env._fake_protocol = protocol
        return env
    except ImportError:
        # Fallback: lightweight replica used when ray/rllib is not installed
        return _StandaloneEnv(protocol)


# ---------------------------------------------------------------------------
# Lightweight replica of RayEnv.step() for standalone / CI use
# ---------------------------------------------------------------------------
class _StandaloneEnv:
    """Reproduces RayEnv.step() tracking logic without requiring ray."""

    def __init__(self, protocol):
        self.protocol = protocol
        self._env_id = 0
        self._terminated_agents = set()
        self._truncated_agents = set()
        self._current_agents = set()

        ids, _, obs_defns, act_defns = protocol.get_definition()
        self.possible_agents = list(ids[0])
        self._current_agents = set(self.possible_agents)
        self._single_action_spaces = act_defns[0]
        self._single_observation_spaces = obs_defns[0]
        self._fake_protocol = protocol

    @property
    def single_action_spaces(self):
        return self._single_action_spaces

    def reset(self, seed=None):
        self._terminated_agents = set()
        self._truncated_agents = set()
        obs, infos = self.protocol.send_reset_msg()
        self._current_agents = set(obs[0].keys())
        return obs[self._env_id], infos[self._env_id]

    def step(self, actions):
        # Forward actions as-is — no no-op padding.
        action_dict = {self._env_id: actions}
        observations, rewards, terminateds, truncateds, infos, _, _ = (
            self.protocol.send_action_msg(action_dict, self._single_action_spaces)
        )

        eid = self._env_id
        all_agents_this_step = set(terminateds[eid]) | set(truncateds[eid])

        for a in all_agents_this_step:
            if terminateds[eid].get(a):
                self._terminated_agents.add(a)
            if truncateds[eid].get(a):
                self._truncated_agents.add(a)

        self._current_agents = {
            a for a in all_agents_this_step
            if not terminateds[eid].get(a) and not truncateds[eid].get(a)
        }

        all_known = self._current_agents | self._terminated_agents | self._truncated_agents
        num_done = len(self._terminated_agents | self._truncated_agents)
        num_total = len(all_known)

        terminateds[eid]["__all__"] = (num_done == num_total) if num_total > 0 else False
        truncateds[eid]["__all__"] = (
            (len(self._truncated_agents) == num_total) if num_total > 0 else False
        )

        return (
            observations[eid],
            rewards[eid],
            terminateds[eid],
            truncateds[eid],
            infos[eid],
        )


# ===========================================================================
# Tests
# ===========================================================================


def test_only_live_agent_actions_forwarded():
    """
    Python must send exactly the actions RLlib provides — no zero-padding.

    The C++ TScholaEnvironment::Step() is responsible for skipping dead agents
    that are absent from InActions.  If Python were to pad dead agents with
    zeros, hierarchical-RL policies that distinguish no-op from a valid action
    would produce incorrect behaviour.
    """
    env = make_ray_env()
    obs, _ = env.reset()
    protocol = env._fake_protocol
    action_spaces = env.single_action_spaces

    # Step 1: all 3 agents alive — all 3 actions sent
    actions = {a: action_spaces[a].sample() for a in obs}
    env.step(actions)
    assert protocol.received_actions_log[0] == {"agent_0", "agent_1", "agent_2"}

    # Step 2: RLlib only provides actions for agent_1 and agent_2 (agent_0 is dead)
    env.step({a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]})
    assert protocol.received_actions_log[1] == {
        "agent_1", "agent_2"
    }, "Dead agent_0 must NOT be zero-padded into the action map"

    # Step 3: only agent_2 alive
    env.step({"agent_2": action_spaces["agent_2"].sample()})
    assert protocol.received_actions_log[2] == {
        "agent_2"
    }, "Only the last living agent's action should reach the protocol"


def test_all_flag_false_while_agents_alive():
    """__all__ must stay False until every agent is terminated/truncated."""
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    # Step 1: agent_0 dies
    _, _, terminateds, truncateds, _ = env.step(
        {a: action_spaces[a].sample() for a in obs}
    )
    assert terminateds["agent_0"] is True
    assert terminateds["__all__"] is False, "agent_1 and agent_2 are still alive"

    # Step 2: agent_1 dies
    _, _, terminateds, truncateds, _ = env.step(
        {a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]}
    )
    assert terminateds["agent_1"] is True
    assert terminateds["__all__"] is False, "agent_2 is still alive"


def test_all_flag_true_when_last_agent_dies():
    """__all__ must become True on the step the last living agent dies."""
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    env.step({a: action_spaces[a].sample() for a in obs})
    env.step({a: action_spaces[a].sample() for a in ["agent_1", "agent_2"]})

    # Step 3: agent_2 dies — all agents now dead
    _, _, terminateds, truncateds, _ = env.step(
        {"agent_2": action_spaces["agent_2"].sample()}
    )
    assert terminateds["agent_2"] is True
    assert terminateds["__all__"] is True, "All agents dead — episode must be marked done"


def test_episode_completes_within_bounded_steps():
    """
    Driving the env with only live agents' actions must reach __all__=True.

    A hang here means either the Python tracking or the C++ terminal-state
    preservation (TScholaEnvironment::Step) is broken.
    """
    env = make_ray_env()
    obs, _ = env.reset()
    action_spaces = env.single_action_spaces

    done = False
    for _ in range(10):
        live = [a for a in obs if a != "__all__"]
        if not live:
            break
        obs, _, terminateds, _, _ = env.step(
            {a: action_spaces[a].sample() for a in live}
        )
        if terminateds.get("__all__"):
            done = True
            break

    assert done, "Episode never reached __all__=True — staggered-death hang detected"


# ===========================================================================
# Standalone runner
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("test_only_live_agent_actions_forwarded", test_only_live_agent_actions_forwarded),
        ("test_all_flag_false_while_agents_alive", test_all_flag_false_while_agents_alive),
        ("test_all_flag_true_when_last_agent_dies", test_all_flag_true_when_last_agent_dies),
        ("test_episode_completes_within_bounded_steps", test_episode_completes_within_bounded_steps),
    ]

    passed = failed = 0
    errors = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if errors:
        for name, e in errors:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
