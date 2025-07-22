<h2>Module Documentation</h2>

<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Description</th>
      <th>Main Methods</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>EnsembleLinear</code></td>
      <td>A fully-connected linear layer, but implemented as a stack of num_ensemble independent layers. Each ensemble member has its own weights and biases.</td>
      <td>
        <ul>
          <li><code>forward(x)</code></li>
          <li><code>load_save()</code></li>
          <li><code>update_save()</code></li>
          <li><code>get_decay_loss()</code></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>EnsembleDynamicsModel</code></td>
      <td>A feedforward ensemble neural network with multiple hidden layers using EnsembleLinear. Output layer of size 2 * (obs_dim + reward_dim): First half is the predicted mean while second half is the predicted log variance (uncertainty).</td>
      <td>
        <ul>
          <li><code>forward(obs_action)</code></li>
          <li><code>load_save()</code></li>
          <li><code>update_save(indexes)</code></li>
          <li><code>get_decay_loss()</code></li>
          <li><code>set_elites(indexes)</code></li>
          <li><code>random_elite_idxs(batch_size)</code></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><code>EnsembleDynamics</code></td>
      <td>Ensemble of neural networks for modeling environment dynamics, supports uncertainty estimation and elite selection.</td>
      <td>
        <ul>
          <li><code>step(obs, action)</code></li>
          <li><code>format_samples_for_training(data)</code></li>
          <li><code>train(data, wandb, logger, max_epochs, max_epochs_since_update, batch_size, holdout_ratio, logvar_loss_coef)</code></li>
          <li><code>learn(inputs, targets, batch_size, logvar_loss_coef)</code></li>
          <li><code>validate(inputs, targets)</code></li>
          <li><code>select_elites(metrics)</code></li>
          <li><code>save(save_path)</code></li>
          <li><code>load(load_path)</code></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
