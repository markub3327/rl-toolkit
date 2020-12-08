from .network import Actor, Critic

import tensorflow as tf

class TD3:
    """
        Twin Delayed DDPG

        https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, 
                 state_shape, 
                 action_shape,
                 learning_rate, 
                 tau,
                 gamma,
                 target_noise,
                 noise_clip,
                 policy_delay,
                 model_a_path,
                 model_c1_path,
                 model_c2_path):

        self._gamma = tf.constant(gamma)
        self._tau = tf.constant(tau)
        self._target_noise = tf.constant(target_noise)
        self._noise_clip = tf.constant(noise_clip)
        self._policy_delay = policy_delay

        # logging metrics
        #tf.keras.metrics.Mean()

        # Actor network & target network
        self.actor = Actor(state_shape, action_shape, learning_rate, model_path=model_a_path)
        self.actor_targ = Actor(state_shape, action_shape, learning_rate, model_path=model_a_path)

        # Critic network & target network
        self.critic_1 = Critic(state_shape, action_shape, learning_rate, model_path=model_c1_path)
        self.critic_targ_1 = Critic(state_shape, action_shape, learning_rate, model_path=model_c1_path)

        # Critic network & target network
        self.critic_2 = Critic(state_shape, action_shape, learning_rate, model_path=model_c2_path)
        self.critic_targ_2 = Critic(state_shape, action_shape, learning_rate, model_path=model_c2_path)

        # first make a hard copy
        self._update_target(self.actor, self.actor_targ, tau=tf.constant(1.0))
        self._update_target(self.critic_1, self.critic_targ_1, tau=tf.constant(1.0))
        self._update_target(self.critic_2, self.critic_targ_2, tau=tf.constant(1.0))

    @tf.function
    def get_action(self, state):
        s = tf.expand_dims(state, axis=0)       # add batch_size=1 dim
        a = self.actor.model(s)
        return tf.squeeze(a, axis=0)            # remove batch_size dim

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.model.trainable_variables, net_targ.model.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

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
        Q_targets = batch['rew'] + (1 - batch['done']) * self._gamma * next_q

        # update critic '1'
        with tf.GradientTape() as tape:
            q_values = self.critic_1.model([batch['obs'], batch['act']])
            q_losses = tf.keras.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q1_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q1_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads, self.critic_1.model.trainable_variables))

        # update critic '2'
        with tf.GradientTape() as tape:
            q_values = self.critic_2.model([batch['obs'], batch['act']])
            q_losses = tf.keras.losses.mean_squared_error(y_true=Q_targets, y_pred=q_values)
            q2_loss = tf.nn.compute_average_loss(q_losses)

        grads = tape.gradient(q2_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads, self.critic_2.model.trainable_variables))

        return q1_loss, q2_loss

    # ------------------------------------ update actor ----------------------------------- #
    @tf.function
    def _update_actor(self, batch):
        with tf.GradientTape() as tape:
            # predict action
            y_pred = self.actor.model(batch['obs'])
            # predict q value
            q_pred = self.critic_1.model([batch['obs'], y_pred])
        
            # compute per example loss
            a_loss = tf.nn.compute_average_loss(-q_pred)

        grads = tape.gradient(a_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))

        return a_loss

    def train(self, batch, t):
        # Critic models update
        loss_c1, loss_c2 = self._update_critic(batch)

        # Delayed policy update
        if (t % self._policy_delay == 0):
            loss_a = self._update_actor(batch)

            # ---------------------------- soft update target networks ---------------------------- #
            self._update_target(self.actor, self.actor_targ, tau=self._tau)
            self._update_target(self.critic_1, self.critic_targ_1, tau=self._tau)
            self._update_target(self.critic_2, self.critic_targ_2, tau=self._tau)
        else:
            loss_a = None

        #print(t, loss_a, loss_c1, loss_c2)

        return loss_a, loss_c1, loss_c2