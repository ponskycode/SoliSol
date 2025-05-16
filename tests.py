from solitaire_env import *

def manual_test(env):
    obs, _ = env.reset()
    render_env(obs)

    max_seq_len = env.max_seq_len
    num_cols = env.num_columns

    while True:
        print("\n--- AVAILABLE ACTIONS ---")
        print("1. Draw from deck → type: `draw`")
        print("2. Move sequence → type: `from to length` (e.g. `2 4 3`)")
        print("   - `from`: source column (0–5)")
        print("   - `to`: target column (0–5)")
        print("   - `length`: sequence length (1–6)")
        print("3. Exit → type: `exit`")

        user_input = input("Your action: ").strip().lower()
        if user_input == "exit":
            break
        elif user_input == "draw":
            action = 0
        else:
            try:
                parts = list(map(int, user_input.split()))
                if len(parts) != 3:
                    print("❌ Enter exactly 3 numbers: from, to, length")
                    continue
                from_col, to_col, length = parts
                if not (0 <= from_col < num_cols and 0 <= to_col < num_cols):
                    print("❌ Invalid column number")
                    continue
                if not (1 <= length <= max_seq_len):
                    print("❌ Sequence length out of range (1–{})".format(max_seq_len))
                    continue

                # Encode action
                seq_len_index = length - 1
                action = 1 + (from_col * num_cols * max_seq_len) + (to_col * max_seq_len) + seq_len_index

            except ValueError:
                print("❌ Invalid input – enter numbers or `draw`")
                continue

        # Perform action
        obs, reward, done, truncated, info = env.step(action)
        render_env(obs)
        print(f"🎯 Action performed: {action}")
        print(f"🎁 Reward: {reward}")
        if done:
            print("🏁 Game over!")
            break
