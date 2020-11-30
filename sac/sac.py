from .network import Actor, Critic

import tensorflow as tf


class SAC:
    """
        Soft Actor-Critic

        https://arxiv.org/pdf/1812.05905.pdf
    """

    def __init__(self, 
                 state_shape, 
                 action_shape,
                 learning_rate, 
                 tau,
                 gamma):

        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self.alpha = tf.constant(0.2)

        # Actor network & target network
        self.actor = Actor(state_shape, action_shape, learning_rate)

        # Critic network & target network
        self.critic_1 = Critic(state_shape, action_shape, learning_rate)
        self.critic_targ_1 = Critic(state_shape, action_shape, learning_rate)

        # Critic network & target network
        self.critic_2 = Critic(state_shape, action_shape, learning_rate)
        self.critic_targ_2 = Critic(state_shape, action_shape, learning_rate)

        # first make a hard copy
        self._update_target(self.critic_1, self.critic_targ_1, tau=tf.constant(1.0))
        self._update_target(self.critic_2, self.critic_targ_2, tau=tf.constant(1.0))

    @tf.function
    def get_action(self, state):
        s = tf.expand_dims(state, axis=0)       # add batch_size=1 dim
        a, _ = self.actor.predict(s, with_logprob=False)
        return tf.squeeze(a, axis=0)            # remove batch_size dim

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.model.trainable_variables, net_targ.model.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function
    def train(self, batch):
        # ------------------------------------ update critic ----------------------------------- #
        next_action, log_a2 = self.actor.predict(batch['obs2'])

        # target Q-values
        q_1 = self.critic_targ_1.model([batch['obs2'], next_action])
        q_2 = self.critic_targ_2.model([batch['obs2'], next_action])
        next_q = tf.minimum(q_1, q_2)

        # Use Bellman Equation! (recursive definition of q-values)
        Q_targets = batch['rew'] + (1 - batch['done']) * self._gamma * (next_q - self.alpha * log_a2)

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self.critic_1.model([batch['obs'], batch['act']])
            q_losses = tf.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q1_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q1_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads, self.critic_1.model.trainable_variables))

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self.critic_2.model([batch['obs'], batch['act']])
            q_losses = tf.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads, self.critic_2.model.trainable_variables))

        # ------------------------------------ update actor ----------------------------------- #
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_a = self.actor.predict(batch['obs'])
            # predict q value
            q1_pred = self.critic_1.model([batch['obs'], y_pred])
            q2_pred = self.critic_2.model([batch['obs'], y_pred])
            q_pred = tf.minimum(q1_pred, q2_pred)

            # compute per example loss
            a_loss = tf.nn.compute_average_loss(self.alpha * log_a - q_pred)

        grads = tape.gradient(a_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))

        # ---------------------------- soft update target networks ---------------------------- #
        self._update_target(self.critic_1, self.critic_targ_1, tau=self._tau)
        self._update_target(self.critic_2, self.critic_targ_2, tau=self._tau)
        
        return a_loss, (q1_loss + q2_loss)