import matplotlib.pyplot as plt

plt.figure(figsize=(6 * 2, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Steps')
    plt.title('Total reward')
    plt.ylabel('Accumulated reward')
    plt.plot(range(len(rewards)), rewards)

    plt.subplot(1, 2, 2)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Reward per step')
    plt.plot(range(len(reward_step)), reward_step)

    plt.show()

