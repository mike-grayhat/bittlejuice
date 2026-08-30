"""PPO plus a supervised state estimator, so the actor has a velocity sense the robot cannot
otherwise provide.

WHY THIS EXISTS, measured rather than assumed. A policy trained the obvious way comes out
open-loop: blindfolding the IMU -- replacing ang_vel and projected_gravity with a constant
"upright and still" in the current frame and all history -- costs it a few percent of forward
speed at most, and sometimes nothing at all. Terrain, slope, larger pushes and a high
termination threshold each remove a real obstacle, and none of them alone produces feedback
behaviour.

The cause is structural. The actor's 33 inputs are ang_vel(3) + projected_gravity(3) +
commands(3) + dof_pos(8) + dof_vel(8) + last_action(8), and the joint channels are the host's own
servo model -- its own commands played back, carrying nothing about load or contact. So a robot
drifting sideways, upright and unrotating, produces LITERALLY ZERO SIGNAL, while tracking_lin_vel
penalises it for exactly that drift. Meanwhile the privileged critic already receives the true
base_lin_vel and is then discarded; nothing transfers that knowledge to the actor.

This closes that loop: an estimator reads what the robot really has (IMU + history + its own
action history), predicts base_lin_vel, and its output is appended to the actor's input.

    obs(33) + history(132) --+--> estimator --> v_hat(3) --+
                             +----------------------------+--> actor --> action(8)

    loss_ppo       = surrogate + value + entropy      (untouched)
    loss_estimator = MSE(v_hat, privileged[0:3])      (separate optimizer)

v_hat is DETACHED before the actor consumes it. The estimator then learns only from its
supervised target, and PPO cannot quietly repurpose it into an arbitrary feature extractor whose
output stops meaning "velocity" -- which would also decalibrate the estimator we deploy.

Modelled on rsl_rl's own `rnd`: an auxiliary module with its own optimizer, its own loss, and its
own checkpoint entries, living inside PPO rather than beside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_obs_groups

import policy_io as pio
from mj_vec_env import MjVecEnv

# Which privileged channels the estimator predicts. Defined against the env's own exported slice
# so the target cannot drift from the privileged layout -- inserting a channel ahead of
# base_lin_vel would otherwise retarget this onto dof_pos in silence.
TARGET_SLICE = MjVecEnv.PRIVILEGED_LIN_VEL_SLICE
TARGET_DIM = TARGET_SLICE.stop - TARGET_SLICE.start

# The observation group the algorithm writes its estimate into. The env declares it (zero-filled,
# see MjVecEnv._alloc_buffers) purely so MLPModel can size the actor input and RolloutStorage has
# somewhere to keep the value that was actually used.
ESTIMATE_GROUP = "estimate"
ESTIMATOR_OBS_SET = "estimator"

# What every EVALUATION entrypoint should pass to runner.load(). rsl_rl's PPO.load reads
# optimizer_state_dict unconditionally when load_cfg is None, so a checkpoint without one is
# unloadable by default -- which is the only reason average_checkpoints.py carries the last
# input's Adam moments into a weight-averaged file it explicitly warns you never to resume
# from. Evaluation does not step an optimizer, so it has no business requiring one. Loading
# weights only also makes optimizer-stripped checkpoints (logs/shipped/) first-class.
EVAL_LOAD_CFG = {"actor": True, "critic": True, "optimizer": False,
                 "iteration": True, "rnd": False}


class EstimatorPolicy(nn.Module):
    """estimator -> concat -> actor, as one callable, for eval and export.

    get_inference_policy() hands callers "the policy", and with an estimator in the loop the
    actor alone is only half of it. Returning this instead means mj_eval, blindfold_test and
    export_policy_mj all exercise the same composition the training loop does, rather than each
    re-implementing the injection and being free to disagree with it.
    """

    def __init__(self, estimator: nn.Module, actor: nn.Module) -> None:
        super().__init__()
        self.estimator = estimator
        self.actor = actor

    def forward(self, obs, **kwargs):
        obs[ESTIMATE_GROUP] = self.estimator(obs).detach()
        return self.actor(obs, **kwargs)

    def __getattr__(self, name):
        # Forward the MLPModel surface callers expect (get_output_log_prob, reset, ...) to the
        # actor, so this is a drop-in for the model get_policy used to return.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["actor"], name)


class PPOWithEstimator(PPO):
    """PPO with a concurrently-trained supervised state estimator."""

    def __init__(self, *args, estimator_cfg: dict | None = None,
                 lipschitz_coef: float = 0.0, symmetry_coef: float = 0.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # LCP (arXiv:2410.11825). 0.0 disables, so every existing run is unaffected.
        self.lipschitz_coef = float(lipschitz_coef)
        self.symmetry_coef = float(symmetry_coef)
        # Set by mj_train when the penalty curriculum is on, so the symmetry loss can ride
        # the same feedback controller the reward penalties do. See _symmetry_penalty.
        self.symmetry_env = None
        # Only the channels that come from SENSORS. "estimate" is the estimator's own
        # detached output, and constraining sensitivity to it would regularise the wrong
        # map -- the estimator is already trained supervised.
        self._lipschitz_obs_keys = ("policy", "history")
        cfg = dict(estimator_cfg or {})
        obs = cfg.pop("obs")
        obs_groups = cfg.pop("obs_groups")

        self.estimator = MLPModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=ESTIMATOR_OBS_SET,
            output_dim=TARGET_DIM,
            hidden_dims=cfg.pop("hidden_dims", [256, 128]),
            activation=cfg.pop("activation", "elu"),
        ).to(self.device)
        self.estimator_optimizer = torch.optim.Adam(
            self.estimator.parameters(), lr=float(cfg.pop("learning_rate", 1e-3))
        )
        print(f"Estimator Model: {self.estimator}")

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> PPO:
        """Resolve the estimator's observation set, then build exactly as PPO does.

        `obs` and the resolved groups are threaded through `estimator_cfg` because PPO's own
        __init__ signature takes prebuilt models and never sees them.
        """
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if ESTIMATOR_OBS_SET not in cfg["obs_groups"]:
            raise KeyError(
                f"obs_groups has no '{ESTIMATOR_OBS_SET}' set. The estimator must be told which "
                f"groups it may read -- and it must read only groups the ROBOT has, or it learns "
                f"to estimate velocity from information the hardware cannot supply."
            )
        if ESTIMATE_GROUP not in obs.keys():
            raise KeyError(
                f"the environment emits no '{ESTIMATE_GROUP}' group; set env_cfg['estimator_dim']"
            )
        est = cfg["algorithm"].setdefault("estimator_cfg", {})
        est["obs"], est["obs_groups"] = obs, cfg["obs_groups"]
        return PPO.construct_algorithm(obs, env, cfg, device)

    # -- rollout ------------------------------------------------------------------------
    def act(self, obs):
        """Fill the estimate group, then act exactly as PPO does.

        Writing it into `obs` BEFORE super().act means the stored transition carries the same
        estimate the actor actually saw, so the update's re-forward is consistent with the
        rollout. Storing a different value would make PPO's probability ratio compare two
        different policies -- the same class of inconsistency that produced this project's
        phantom "latency cliff", where the plant got a delayed action but the observation
        carried the undelayed one.
        """
        obs[ESTIMATE_GROUP] = self.estimator(obs).detach()
        return super().act(obs)

    # -- learning -----------------------------------------------------------------------
    def _lipschitz_penalty(self) -> dict[str, float]:
        """LCP: penalise the gradient of log pi(a|s) with respect to the OBSERVATION.

            max_pi J(pi) - lambda_gp * E[ ||grad_s log pi(a|s)||^2 ]      (arXiv:2410.11825)

        Why this and not another reward term: every smoothness penalty in reward_scales is
        TEMPORAL -- it compares the action now to the action a step ago. None of them ask
        whether a SMALL CHANGE IN OBSERVATION produces a small change in action, which is
        exactly how IMU noise and servo-model error become action jitter, and jitter is what
        a direction reversal is. This matters because the smoothness penalties interact:
        four hand-tuned terms on one objective, and pushing any of them distorts the others.
        Both CAPS
        (arXiv:2012.06644) and LCP report this approach does not scale.

        DEVIATION FROM THE PAPER, stated because it matters: LCP adds the penalty to the PPO
        loss and takes one combined step. Doing that here would mean copying rsl_rl's entire
        update() to insert one term, so this takes a SEPARATE gradient step on the penalty
        alone, alternating with PPO's. That is a different optimisation, closer to how
        decoupled weight decay relates to L2. If the result is ambiguous, this is the first
        thing to suspect.

        Second-order: the penalty is itself a gradient, so create_graph=True is required or
        it has no effect on the actor's weights at all -- it would compute a number and
        backpropagate nothing.
        """
        if not self.lipschitz_coef:
            return {}
        total, batches = 0.0, 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,
                                                       self.num_learning_epochs):
            obs = {k: v for k, v in batch.observations.items()}
            leaves = []
            for k in self._lipschitz_obs_keys:
                if k in obs:
                    obs[k] = obs[k].detach().clone().requires_grad_(True)
                    leaves.append(obs[k])
            if not leaves:
                return {}
            self.actor(obs, masks=batch.masks, hidden_state=batch.hidden_states[0],
                       stochastic_output=True)
            logp = self.actor.get_output_log_prob(batch.actions)
            grads = torch.autograd.grad(logp.sum(), leaves, create_graph=True)
            pen = sum(g.pow(2).sum(dim=-1) for g in grads).mean()

            self.optimizer.zero_grad()
            (self.lipschitz_coef * pen).backward()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += pen.item()
            batches += 1
        return {"lipschitz_grad_sq": total / max(batches, 1)}

    def _symmetry_penalty(self) -> dict[str, float]:
        """Mirror-symmetry loss: the mirrored world should produce the mirrored action.

            L_sym = E[ || pi(s) - M_a( pi( M_s(s) ) ) ||^2 ]

        Yu, Turk, Liu (2018), arXiv:1801.08093, who pair it with an energy penalty for
        exactly the reason it is needed here -- minimising energy alone buys asymmetric
        gaits. Abdolhosseini et al. (2019) compare four ways to enforce symmetry (network
        architecture, data duplication, this loss, phase mirroring) and find the LOSS the
        most consistent and duplication the least effective, which is why this is a loss and
        not a reward term or an augmented batch.

        MEASURED BEFORE BUILDING. The residual above, normalised by ||pi(s)||, reads 45-48%
        on trained policies. Nothing has ever asked them to be symmetric, so they are not, and
        the mirrored action differs from the mirror of the action by about half the action's
        own magnitude. There is real headroom here.

        The asymmetry is not always the one you can see. Sometimes it shows up as POSE -- one
        front shoulder sitting 11 deg further forward than the other -- and sometimes it hides
        in AMPLITUDE, one knee sweeping 12 deg against the other's 27 from the same mean. A
        metric built on mean pose alone scores the second case clean and misses it entirely;
        the residual catches both, because it compares the policy rather than the posture.

        The mirror map itself is validated against MuJoCo rather than against a policy: put
        the robot in a mirrored physical state and every derived feature comes back mirrored
        (tests/test_mirror_symmetry.py). That check matters because a wrong map does not
        crash -- it trains against a target that is not the mirror of anything, and the run
        returns looking like symmetry did not help.

        Same deviation from the paper as _lipschitz_penalty, and for the same reason: a
        separate gradient step rather than a term folded into PPO's loss.
        """
        if not self.symmetry_coef:
            return {}
        # RIDE THE PENALTY CURRICULUM. Measured at 300 iterations, mean reward and episode
        # length against the coefficient:
        #
        #     0.00   reward  1.92   ep len 780
        #     0.03   reward -0.10   ep len 504
        #     0.30   reward -1.54   ep len 499
        #
        # Even 0.03 halves episode length before the policy can walk, which is the same
        # failure `power` has and for the same reason: a policy that cannot walk yet can
        # always satisfy a symmetry penalty by doing nothing, and standing still is
        # perfectly symmetric. Abdolhosseini et al. (2019) note symmetry enforcement can
        # slow learning; here it prevents it outright.
        #
        # Deliberately the env's FEEDBACK factor rather than an iteration schedule. A
        # timetable cannot notice it has broken the policy, which is the lesson
        # CURRICULUM_EP_LEN_LO was written down for. Falls back to 1.0 when there is no
        # curriculum, so --symmetry-coef without --penalty-curriculum behaves as before.
        factor = 1.0
        env = self.symmetry_env
        if env is not None and getattr(env, "penalty_curriculum", False):
            factor = float(getattr(env, "_curriculum_factor", 1.0))
        if factor <= 0.0:
            return {"symmetry_sq": 0.0, "symmetry_factor": 0.0}
        coef = self.symmetry_coef * factor
        total, batches = 0.0, 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,
                                                       self.num_learning_epochs):
            obs = batch.observations
            # output_mean, not a sampled action: symmetry is a property of the POLICY, and
            # comparing two samples would measure exploration noise, which is symmetric in
            # expectation and would drive the penalty toward a constant rather than toward
            # a symmetric mean.
            self.actor(obs, masks=batch.masks, hidden_state=batch.hidden_states[0],
                       stochastic_output=True)
            action = self.actor.output_mean
            self.actor(pio.mirror_obs_groups(obs), masks=batch.masks,
                       hidden_state=batch.hidden_states[0], stochastic_output=True)
            mirrored = pio.mirror_action(self.actor.output_mean)
            pen = (action - mirrored).pow(2).sum(dim=-1).mean()

            self.optimizer.zero_grad()
            (coef * pen).backward()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += pen.item()
            batches += 1
        return {"symmetry_sq": total / max(batches, 1), "symmetry_factor": factor}

    def update(self) -> dict[str, float]:
        """Estimator, then the Lipschitz penalty, then PPO.

        Order is load-bearing: PPO.update() ends with self.storage.clear() (ppo.py:342), so
        anything that iterates the rollout must run BEFORE it. Running the estimator
        afterwards would train on an emptied buffer and report a clean zero loss for the
        whole run.
        """
        stats = self._update_estimator()
        stats.update(self._lipschitz_penalty())
        stats.update(self._symmetry_penalty())
        stats.update(super().update())
        return stats

    def _update_estimator(self) -> dict[str, float]:
        total, batches, sq_err, sq_tgt = 0.0, 0, 0.0, 0.0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,
                                                       self.num_learning_epochs):
            target = batch.observations["privileged"][:, TARGET_SLICE]
            pred = self.estimator(batch.observations)
            loss = (pred - target).pow(2).mean()

            self.estimator_optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            total += loss.item()
            batches += 1
            sq_err += (pred - target).pow(2).sum().item()
            sq_tgt += (target - target.mean(dim=0, keepdim=True)).pow(2).sum().item()

        if batches == 0:
            return {"estimator_loss": float("nan"), "estimator_r2": float("nan")}
        # R^2 AGAINST THE MEAN, not against zero.
        #
        # This started as nrmse = ||err|| / ||target||, whose "no better than chance" point is
        # 1.0. That bar is far too lenient: measured on a trained estimator, simply
        # predicting the CONSTANT MEAN velocity scores 0.585 there while
        # carrying no information at all, because forward velocity is dominated by its mean.
        # The trained estimator scored 0.430 -- which sounds like most of the way to perfect and
        # is actually only 46% of the variance ABOUT the mean.
        #
        # R^2 puts the useless predictor at exactly 0.0, which is what a threshold needs.
        r2 = 1.0 - sq_err / sq_tgt if sq_tgt > 0 else float("nan")
        return {"estimator_loss": total / batches, "estimator_r2": r2}

    # -- plumbing -----------------------------------------------------------------------
    def get_policy(self):
        return EstimatorPolicy(self.estimator, self._raw_actor)

    def train_mode(self) -> None:
        super().train_mode()
        self.estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.estimator.eval()

    def save(self) -> dict:
        saved = super().save()
        saved["estimator_state_dict"] = self.estimator.state_dict()
        saved["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        if "estimator_state_dict" in loaded_dict:
            self.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=strict)
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator_optimizer.load_state_dict(
                    loaded_dict["estimator_optimizer_state_dict"])
        elif strict:
            raise KeyError(
                "checkpoint has no estimator_state_dict -- it was trained without --estimator, "
                "so its actor expects a narrower observation than this config builds"
            )
        return resumed
