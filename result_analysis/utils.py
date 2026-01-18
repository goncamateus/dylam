from matplotlib import font_manager, pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from yaml import safe_load


METHOD_COLORS = {
    "Q-Learning": "#FF6F00",
    "DQN": "#FF6F00",
    "SAC": "#10FF00",
    "Baseline": "#FF6F00",
    "DyLam": "#10FF00",
    "drQ": "#0091FF",
    "Q-Decomp": "#EE00FF",
    "Q-Decomposition": "#EE00FF",
    "DRQ": "#0091FF",
    "Tuned-DRQ": "#FFD900",
    "GPILS": "#EE00FF",
}

FORMATTER = ticker.ScalarFormatter(useMathText=True)
FORMATTER.set_scientific(True)
FORMATTER.set_powerlimits((-1, 1))
FONT = font_manager.FontProperties(weight='bold')

PARAMS = safe_load(open("../scripts/experiments.yml", "r"))
REWARD_RANGES = {
    value["gym_id"].replace("mo-", ""): {
        "r_max": value["r_max"],
        "r_min": value["r_min"],
    }
    for value in PARAMS["Dylam"].values()
}


def smooth_curve(points: np.ndarray, factor: int = 100):
    cumsum = np.cumsum(np.insert(points, 0, 0))
    return (cumsum[factor:] - cumsum[:-factor]) / float(factor)


def plot_result(
    gym_id,
    results: dict,
    formatter: ticker.ScalarFormatter,
    colors: dict,
    y_label: str,
    x_label: str = "Number of training steps",
    smooth_factor: int = 100,
    smooth_factor_min_max: int = 100,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in results.keys():
        mean_key = [
            key
            for key in results[method].keys()
            if not (
                key.startswith("Step") or key.endswith("MAX") or key.endswith("MIN")
            )
        ][0]
        min_key = [key for key in results[method].keys() if key.endswith("MIN")][0]
        max_key = [key for key in results[method].keys() if key.endswith("MAX")][0]

        x = results[method]["Step"].loc[smooth_factor - 1 :]
        y = smooth_curve(results[method][mean_key], factor=smooth_factor)
        ax.plot(x, y, label=method, color=colors[method])

        x = results[method]["Step"].loc[smooth_factor_min_max - 1 :]
        y_min = smooth_curve(results[method][min_key], smooth_factor_min_max)
        y_max = smooth_curve(results[method][max_key], smooth_factor_min_max)
        ax.fill_between(x, y_min, y_max, color=colors[method], alpha=0.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=1)
    ax.grid(True)
    ax.legend(prop=FONT)
    ax.set_title(f"{gym_id}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(f"{gym_id}.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def plot_result_taxi(
    gym_id,
    results: dict,
    formatter: ticker.ScalarFormatter,
    colors: dict,
    y_label: str,
    x_label: str = "Number of training steps",
    smooth_factor: int = 100,
):
    def get_reward(result, keys):
        reward = (
            result[keys[0]]
            - abs(result[keys[1]] / 200)
            - abs((result[keys[2]]) / 200)
            + 0.1
        )
        reward = np.clip(reward, -1, 1)
        return reward

    fig, ax = plt.subplots()
    for method in results.keys():
        mean_keys = [
            key
            for key in results[method].keys()
            if not (
                key.startswith("Step") or key.endswith("MAX") or key.endswith("MIN")
            )
        ]
        min_keys = [key for key in results[method].keys() if key.endswith("MIN")]
        max_keys = [key for key in results[method].keys() if key.endswith("MAX")]

        result_mean = get_reward(results[method], mean_keys)
        result_min = get_reward(results[method], min_keys)
        result_max = get_reward(results[method], max_keys)

        x = results[method]["Step"].loc[smooth_factor - 1 :]
        y = smooth_curve(result_mean, factor=smooth_factor)
        y_min = smooth_curve(result_min, factor=smooth_factor)
        y_max = smooth_curve(result_max, factor=smooth_factor)

        # Plot the data using y as the mean continuous line and y_min/y_max as shaded regions
        ax.plot(x, y, label=method, color=colors[method])
        ax.fill_between(x, y_min, y_max, color=colors[method], alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=1)
    ax.grid(True)
    ax.legend(prop=FONT)
    ax.set_title(f"{gym_id}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(f"{gym_id}.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def plot_lambdas(
    title, lambdas, formatter, smooth_factor, x_label="Number of training steps"
):
    COLOR = {
        0: "red",
        1: "blue",
        2: "green",
        3: "orange",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, weight in enumerate(lambdas.keys()):
        mean_key = [
            key
            for key in lambdas[weight].keys()
            if not (
                key.startswith("Step") or key.endswith("MAX") or key.endswith("MIN")
            )
        ][0]

        x = lambdas[weight]["Step"].loc[smooth_factor - 1 :]
        y = smooth_curve(lambdas[weight][mean_key], factor=smooth_factor)
        ax.plot(x, y, label=weight.replace("_", " "), color=COLOR[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\lambda$ weights")
    ax.set_ylim(0, 1)
    ax.grid(True)
    # ax.legend(prop=FONT)
    ax.set_title(f"{title}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    path = title + "-weights.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_rewards(
    title,
    rewards,
    reward_max,
    reward_min,
    formatter,
    smooth_factor,
    x_label="Number of training steps",
):
    def normalize_rewards(rew, reward_max, reward_min):
        rew = np.array(rew)
        abs_max = max(abs(reward_max), abs(reward_min))
        rew = rew / abs_max
        return rew

    PARAM = {
        0: ("red", (0, (3, 1))),
        1: ("blue", (0, (3, 4))),
        2: ("green", (0, (3, 8))),
        3: ("orange", (0, (3, 12))),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    normalized_max_rewards = [
        normalize_rewards([reward_max[i]], reward_max[i], reward_min[i])
        for i in range(len(reward_max))
    ]
    for i, reward in enumerate(rewards.keys()):
        mean_key = [
            key
            for key in rewards[reward].keys()
            if not (
                key.startswith("Step") or key.endswith("MAX") or key.endswith("MIN")
            )
        ][0]

        x = rewards[reward]["Step"].loc[smooth_factor - 1 :]
        r_max_line = [normalized_max_rewards[i]] * len(x)
        ax.plot(x, r_max_line, color=PARAM[i][0], linestyle=PARAM[i][1])
        y = smooth_curve(rewards[reward][mean_key], factor=smooth_factor)
        y = normalize_rewards(y, reward_max[i], reward_min[i])
        ax.plot(x, y, label=reward.replace("_", " "), color=PARAM[i][0])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Episode Rewards", labelpad=1)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)
    # ax.legend(prop=FONT)
    ax.set_title(title)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    path = title + "-components.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def pareto_frontier(obj1, obj2):
    """
    Find the indices of the Pareto frontier in a two-objective optimization problem.
    
    Parameters:
        obj1 (np.ndarray): The first objective values.
        obj2 (np.ndarray): The second objective values.
    Returns:
        list: Indices of the Pareto optimal solutions.
    """
    obj_hold = []
    for i, value in enumerate(obj1):
        exists = not (obj1 > value).any()
        exists_obj2 = not (obj2[obj1 > value] > obj2[obj1 == value]).any()
        if exists or exists_obj2:
            obj_hold.append(i)
    return obj_hold