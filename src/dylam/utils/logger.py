import os
import time

import wandb


class WandbResultLogger:
    project = "DyLam"

    def __init__(self, name, params):
        if params.seed == 0:
            params.seed = int(time.time())
        self.run = wandb.init(
            project=self.project,
            name=name,
            entity="goncamateus",
            config=vars(params),
            monitor_gym=True,
            mode=None if params.track else "disabled",
            save_code=True,
        )
        self.comp_names = params.comp_names
        self.artifact = wandb.Artifact("model", type="model")
        self.log = {}
        self._episode = 0

    def log_episode(self, infos, rewards, dones):
        for idx, done in enumerate(dones):
            if done:
                info = infos["final_info"][idx]
                if "Original_reward" not in info:
                    info["Original_reward"] = info["episode"]["r"]
                print(
                    f"Episode {self._episode}: episodic_return={info['Original_reward']}"
                )
                self.log.update(
                    {
                        "ep_info/total": info["Original_reward"],
                    }
                )
                keys_to_log = [x for x in info.keys() if x.startswith("reward_")]
                for key in keys_to_log:
                    self.log[f"ep_info/{key.replace('reward_', '')}"] = info[key]
                self._episode += 1
                break
        self.log.update({"rewards": rewards})

    def log_losses(self, losses):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def log_lambdas(self, lambdas):
        for i in range(len(lambdas)):
            self.log.update({f"lambdas/{self.comp_names[i]}": lambdas[i].item()})

    def log_artifact(self): ...

    def push(self, global_step):
        self.run.log(self.log, global_step)
        self.log = {}


class SACLogger(WandbResultLogger):
    def __init__(self, name, params):
        params.method = "sac"
        super().__init__(name, params)

    def log_losses(self, losses):
        self.log.update(
            {
                "losses/Value1_loss": losses["qf1_loss"].item(),
                "losses/Value2_loss": losses["qf2_loss"].item(),
                "losses/alpha": losses["alpha"],
            }
        )
        if losses["policy_loss"] is not None:
            self.log.update({"losses/policy_loss": losses["policy_loss"].item()})
            if losses["alpha_loss"] is not None:
                self.log.update({"losses/alpha_loss": losses["alpha_loss"].item()})
        if "ori_qf1_loss" in losses:
            self.log.update(
                {
                    "losses/Original_Value1_loss": losses["ori_qf1_loss"].item(),
                    "losses/Original_Value2_loss": losses["ori_qf2_loss"].item(),
                }
            )

    def log_artifact(self):
        self.artifact.add_file(f"models/{self.run.name}/actor.pt")


class QLogger(WandbResultLogger):
    def __init__(self, name, params):
        params.method = "Q-Learning"
        super().__init__(name, params)

    def log_episode(self, info, done):
        if done:
            print(f"Episode {self._episode}: episodic_return={info['Original_reward']}")
            self.log.update(
                {
                    "ep_info/total": info["Original_reward"],
                }
            )
            keys_to_log = [x for x in info.keys() if x.startswith("reward_")]
            for key in keys_to_log:
                self.log[f"ep_info/{key.replace('reward_', '')}"] = info[key]
            self._episode += 1

    def log_artifact(self):
        if self.run.config["track"]:
            for file in os.listdir(f"models/{self.run.name}"):
                if file.endswith(".npy"):
                    self.artifact.add_file(f"models/{self.run.name}/{file}")

    def log_losses(self, losses):
        if len(losses) > 1:
            for i in range(len(losses)):
                self.log.update({f"losses/qf_update_{i}": losses[f"qf_update_{i}"]})
        else:
            self.log.update({"losses/qf_update": losses["qf_update"]})


class DQNLogger(WandbResultLogger):
    def __init__(self, name, params):
        params.method = "dqn"
        super().__init__(name, params)

    def log_losses(self, losses):
        if "qf_loss" in losses:
            self.log.update({"losses/qf_loss": losses["qf_loss"]})

        if "qf_loss_0" in losses:
            for i in range(len(losses)):
                self.log.update({f"losses/qf_loss_{i}": losses[f"qf_loss_{i}"]})

    def log_artifact(self):
        self.artifact.add_file(f"models/{self.run.name}/q_network.pt")


class PPOLogger(WandbResultLogger):
    def __init__(self, name, params):
        params.method = "ppo"
        super().__init__(name, params)

    def log_losses(self, losses):
        self.log.update(
            {
                "losses/loss": losses["loss"].item(),
                "losses/Policy_loss": losses["policy_loss"].item(),
                "losses/Value_loss": losses["qf_loss"].item(),
                "losses/Entropy_loss": losses["entropy_loss"].item(),
            }
        )

    def log_artifact(self):
        self.artifact.add_file(f"models/{self.run.name}/actor.pt")
