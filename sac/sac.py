from .network import Actor, Critic

import tensorflow as tf
import tensorflow_probability as tfp


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

        # init param 'alpha' - Lagrangian
        self._log_alpha = tf.Variable(0.0, trainable=True)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        self._alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='alpha_optimizer')
        self._target_entropy = tf.cast(-tf.reduce_prod(action_shape), dtype=tf.float32)
        print(self._target_entropy)
        print(self._alpha)

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
        a, logp = self.actor.predict(s, with_logprob=True)
        return tf.squeeze(a, axis=0), logp            # remove batch_size dim

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.model.trainable_variables, net_targ.model.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    # ------------------------------------ update critic ----------------------------------- #
    @tf.function
    def _update_critic(self, batch):
        next_action, next_log_prob = self.actor.predict(batch['obs2'])

        # target Q-values
        next_q_1 = self.critic_targ_1.model([batch['obs2'], next_action])
        next_q_2 = self.critic_targ_2.model([batch['obs2'], next_action])
        next_q = tf.minimum(next_q_1, next_q_2)
        #tf.print(f'nextQ: {next_q.shape}')

        # Use Bellman Equation! (recursive definition of q-values)
        Q_targets = tf.stop_gradient(batch['rew'] + (1 - batch['done']) * self._gamma * (next_q - self._alpha * next_log_prob))
        #tf.print(f'qTarget: {Q_targets.shape}')

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self.critic_1.model([batch['obs'], batch['act']])
            q_losses = 0.5 * tf.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q1_loss = tf.nn.compute_average_loss(q_losses)
        #    tf.print(f'q_val: {q_values.shape}')

        grads = tape.gradient(q1_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads, self.critic_1.model.trainable_variables))

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self.critic_2.model([batch['obs'], batch['act']])
            q_losses = 0.5 * tf.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads, self.critic_2.model.trainable_variables))

        return (q1_loss + q2_loss)

    # ------------------------------------ update actor ----------------------------------- #
    @tf.function
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred, log_prob = self.actor.predict(batch['obs'])
            # predict q value
            q_1 = self.critic_1.model([batch['obs'], y_pred])
            q_2 = self.critic_2.model([batch['obs'], y_pred])
            q = tf.minimum(q_1, q_2)
        #    tf.print(f'q: {q.shape}')

            a_losses = self._alpha * log_prob - q
            a_loss = tf.nn.compute_average_loss(a_losses)
        #    tf.print(f'a_losses: {a_losses}')

        grads = tape.gradient(a_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))

        return a_loss

    # ------------------------------------ update alpha ----------------------------------- #
    @tf.function
    def _update_alpha(self, batch):
        y_pred, log_prob = self.actor.predict(batch['obs'])
        #tf.print(f'y_pred: {y_pred.shape}')
        #tf.print(f'log_prob: {log_prob.shape}')
        
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (self._log_alpha * tf.stop_gradient(log_prob + self._target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        #    tf.print(f'alpha_losses: {alpha_losses.shape}')

        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))

        return alpha_loss

    @tf.function
    def train(self, batch):
        q_loss = self._update_critic(batch)
        p_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)

        # ---------------------------- soft update target networks ---------------------------- #
        self._update_target(self.critic_1, self.critic_targ_1, tau=self._tau)
        self._update_target(self.critic_2, self.critic_targ_2, tau=self._tau)
        
        return p_loss, q_loss, alpha_loss, self._alpha