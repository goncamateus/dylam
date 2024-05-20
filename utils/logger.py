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
        self.artifact = wandb.Artifact("model", type="model")
        self.log = {}

    def log_episode(self, infos, rewards):
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info:
                    if "episode" in info:
                        print(f"episodic_return={info['Original_reward']}")
                        self.log.update(
                            {
                                "ep_info/total": info["Original_reward"],
                            }
                        )
                        keys_to_log = [
                            x for x in info.keys() if x.startswith("reward_")
                        ]
                        for key in keys_to_log:
                            self.log[f"ep_info/{key.replace('reward_', '')}"] = info[
                                key
                            ]
                    break
        self.log.update({"rewards": rewards})

    def log_losses(self, losses):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def log_lambdas(self, lambdas):
        for i in range(len(lambdas)):
            self.log.update({"lambdas/component_" + str(i): lambdas[i].item()})

    def log_artifact(self):
        self.artifact.add_file(f"models/{self.run.name}/actor.pt")

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


class DDPGLogger(WandbResultLogger):
    def __init__(self, name, params):
        params.method = "ddpg"
        super().__init__(name, params)

    def log_losses(self, losses):
        self.log.update(
            {
                "losses/Value_loss": losses["qf_loss"].item(),
            }
        )
        if losses["policy_loss"] is not None:
            self.log.update({"losses/policy_loss": losses["policy_loss"].item()})
