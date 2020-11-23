from .network import Actor, Critic

import tensorflow as tf

class TD3:

    def __init__(self, 
                 state_shape, 
                 action_shape,
                 learning_rate: float,
                 tau: float,
                 gamma: float,
                 target_noise: float,
                 noise_clip: float,
                 policy_delay: int):

        self._gamma = gamma
        self._tau = tau
        self._target_noise = target_noise
        self._noise_clip = noise_clip
        self._policy_delay = policy_delay

        # Actor network & target network
        self.actor = Actor(state_shape, action_shape, learning_rate)
        self.actor_targ = Actor(state_shape, action_shape, learning_rate)

        # Critic network & target network
        self.critic_1 = Critic(state_shape, action_shape, learning_rate)
        self.critic_targ_1 = Critic(state_shape, action_shape, learning_rate)

        # Critic network & target network
        self.critic_2 = Critic(state_shape, action_shape, learning_rate)
        self.critic_targ_2 = Critic(state_shape, action_shape, learning_rate)

        # first make a hard copy
        self._target_update(self.actor, self.actor_targ, 1.0)
        self._target_update(self.critic_1, self.critic_targ_1, 1.0)
        self._target_update(self.critic_2, self.critic_targ_2, 1.0)

    @tf.function
    def get_action(self, state):
        s = tf.expand_dims(state, axis=0)       # add batch_size=1 dim
        a = self.actor.model(s)                 # predict

        return a[0]                             # remove batch_size dim

    @tf.function
    def _target_update(self, net, net_targ, tau):
        for p, p_targ in zip(net.model.trainable_weights, net_targ.model.trainable_weights):
            p_targ.assign(tau * p + (1.0 - tau) * p_targ)       # soft update weights

    # ------------------------------------ update critic ----------------------------------- #
    @tf.function
    def _update_critic(self, batch):
        next_action = self.actor_targ.model(batch['obs2'])

        # target policy smoothing
        epsilon = tf.random.normal(next_action.shape, mean=0.0, stddev=self._target_noise)
        epsilon = tf.clip_by_value(epsilon, -self._noise_clip, self._noise_clip)
        next_action = tf.clip_by_value(next_action + epsilon, -1.0, 1.0)

        # target Q-values
        q_1 = self.critic_targ_1.model([batch['obs2'], next_action])
        q_2 = self.critic_targ_2.model([batch['obs2'], next_action])
        next_q = tf.minimum(q_1, q_2) 

        # Use Bellman Equation! (recursive definition of q-values)
        targets = batch['rew'] + (1 - batch['done']) * self._gamma * next_q

        # update critic '1'
        with tf.GradientTape() as tape1:
            q_pred = self.critic_1.model([batch['obs'], batch['act']], training=True)
            loss_c1 = self.critic_1.loss(targets, q_pred)

        critic_grads = tape1.gradient(loss_c1, self.critic_1.model.trainable_weights)
        self.critic_1.optimizer.apply_gradients(zip(critic_grads, self.critic_1.model.trainable_weights))

        # update critic '2'
        with tf.GradientTape() as tape2:
            q_pred = self.critic_2.model([batch['obs'], batch['act']], training=True)
            loss_c2 = self.critic_2.loss(targets, q_pred)

        critic_grads = tape2.gradient(loss_c2, self.critic_2.model.trainable_weights)
        self.critic_2.optimizer.apply_gradients(zip(critic_grads, self.critic_2.model.trainable_weights))

        return (loss_c1 + loss_c2)

    # ------------------------------------ update actor ----------------------------------- #
    @tf.function
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred = self.actor.model(batch['obs'], training=True)
            # predict q value
            q_pred = self.critic_1.model([batch['obs'], y_pred], training=False)
        
            # compute per example loss
            loss_a = -tf.reduce_mean(q_pred)

        actor_grads = tape.gradient(loss_a, self.actor.model.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_weights))
        
        # ---------------------------- soft update target networks ---------------------------- #
        self._target_update(self.actor, self.actor_targ, self._tau)
        self._target_update(self.critic_1, self.critic_targ_1, self._tau)
        self._target_update(self.critic_2, self.critic_targ_2, self._tau)

        return loss_a

    def train(self, batch, t):
        # Critic models update
        loss_c = self._update_critic(batch)

        # Delayed policy update
        if (t % self._policy_delay == 0):
            loss_a = self._update_actor(batch)
        else:
            loss_a = None

        return loss_a, loss_c