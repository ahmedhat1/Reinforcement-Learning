from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):
        s, _ = self.env.reset()
        a = self.action(s)  # Epsilon-greedy action for initial state

        for i in range(n_timesteps):
            s_, r, terminated, _, _ = self.env.step(a)  # Observe the next state and the reward
            a_ = self.action(s_)  # Select action following the epsilon-greedy policy (-> on-policy)
            self.update_Q(s, a, r, s_, a_)  # You will have to call self.update_Q(...) at every step
            s, a = s_, a_  # Update state and action
            if terminated:  # Do not forget to reset the environment and update the action if you receive a 'terminated' signal
                s, _ = self.env.reset()
                a = self.action(s)

    def update_Q(self, s, a, r, s_, a_):
        # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a Q(s',a') - Q(s,a))
        self.Q[*s, a] = self.Q[*s, a] + self.lr * (r + self.g * self.Q[*s_, a_] - self.Q[*s, a])
