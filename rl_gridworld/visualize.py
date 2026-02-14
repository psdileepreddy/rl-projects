import numpy as np

def print_policy(env, Q):
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    print("\nLearned Policy:")
    for r in range(env.rows):
        row_symbols = []
        for c in range(env.cols):
            s = (r, c)

            if s == env.obstacle:
                row_symbols.append("X")
                continue
            if s == env.goal_state:
                row_symbols.append("G")
                continue

            s_idx = env.state_to_index(s)
            best_a = int(np.argmax(Q[s_idx]))

            if s == env.start_state:
                row_symbols.append("S" + arrows[best_a])
            else:
                row_symbols.append(arrows[best_a])

        print("  ".join(row_symbols))

def print_greedy_path(env, Q, max_steps=20):
    state = env.reset()
    path = [state]

    print("\nGreedy path from Start to Goal:")
    for step in range(max_steps):
        s_idx = env.state_to_index(state)
        action = int(np.argmax(Q[s_idx]))

        next_state, reward, done = env.step(action)
        print(f"Step {step}: {state} -> action {action} -> {next_state} (r={reward})")

        path.append(next_state)
        state = next_state

        if done:
            print("Reached the goal!")
            break

    print("Path:", path)
