import gymnasium as gym

# 環境の作成
env = gym.make("CartPole-v1", render_mode="human")

# 環境のリセットと初期観測の取得
observation, info = env.reset(seed=42)

# エピソードのループ
for _ in range(1000):
    # ランダムなアクションの選択（実際のAIならここで方策を使用）
    action = env.action_space.sample()

    # 環境を1ステップ進める
    observation, reward, terminated, truncated, info = env.step(action)

    # エピソードが終了したら環境をリセット
    if terminated or truncated:
        observation, info = env.reset()

# 環境を閉じる
env.close()
