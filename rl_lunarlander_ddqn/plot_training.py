import os
import csv
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "training_log.csv")


def moving_average(x, window=50):
    out = []
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out.append(sum(x[start:i + 1]) / (i - start + 1))
    return out


def main():
    if not os.path.exists(LOG_PATH):
        print("Log file not found:", LOG_PATH)
        print("Run train_ddqn.py first to generate training_log.csv")
        return

    episodes = []
    ep_rewards = []
    epsilons = []
    buffer_sizes = []
    eval_eps = []
    eval_vals = []

    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            ep_rewards.append(float(row["episode_reward"]))
            epsilons.append(float(row["epsilon"]))
            buffer_sizes.append(int(row["buffer_size"]))

            ev = row["eval_avg"]
            if ev is not None and ev != "":
                eval_eps.append(int(row["episode"]))
                eval_vals.append(float(ev))

    # Debug lengths (useful if something breaks again)
    print("len episodes:", len(episodes))
    print("len ep_rewards:", len(ep_rewards))
    print("len epsilons:", len(epsilons))
    print("len buffer_sizes:", len(buffer_sizes))
    print("len eval_eps:", len(eval_eps))
    print("len eval_vals:", len(eval_vals))

    smoothed = moving_average(ep_rewards, window=50)

    # 1) Training reward (raw + moving average)
    plt.figure()
    plt.plot(episodes, ep_rewards, alpha=0.3)
    plt.plot(episodes, smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("LunarLander training reward (raw + moving avg)")
    plt.legend(["raw", "avg(50)"])

    # 2) Epsilon schedule
    plt.figure()
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon schedule")

    # 3) Replay buffer growth
    plt.figure()
    plt.plot(episodes, buffer_sizes)
    plt.xlabel("Episode")
    plt.ylabel("Replay buffer size")
    plt.title("Replay buffer growth")

    # 4) Evaluation curve (if available)
    if len(eval_vals) > 1 and len(eval_eps) == len(eval_vals):
        plt.figure()
        plt.plot(eval_eps, eval_vals)
        plt.axhline(200, linestyle="--")
        plt.xlabel("Episode")
        plt.ylabel("Eval avg reward")
        plt.title("Evaluation (greedy) with solved threshold")
        plt.legend(["eval avg", "solved=200"])
    else:
        print("Not enough eval points to plot evaluation curve yet.")

    plt.show()


if __name__ == "__main__":
    main()