# PyTorch-Tic-Tac-Toe (Q & Deep Q)
Playing Tic-Tac-Toe with Q learning and Deep Q learning. Achieve average reward **0.966** against the **random player** and **0.0** against the **optimal player**.

Classes in `tic_env.py`:
---

* Environment
  * **TicTacToeEnv**: Classical Tic-tac-toe game for two players.
  * **QAgent**: an environment for Q learning, support learning from expert and learning by self-practice.
  * **DeepQAgent**: an environment for Deep Q learning, support learning from expert and learning by self-practice.
* Player
  * **OptimalPlayer**: an epsilon-greedy optimal player in Tic-tac-toe.
* Others
  * **Buffer**: replay memory for deep Q learning.
  * **FCN**: neural network in deep Q learning.

Methods in  `Q.ipynb` & `Deep_Q.ipynb`:
---

* Experiments
  * **run_simulation**: support experiments that agent learns how to play Tic-Tac-Toe from experts.
  * **self_practice_simulation**: support experiments that agent learns how to play Tic-Tac-Toe by self-practice.
* Plot
  * **render_figure_widge**: basic preparation for creating a figure.
  * **init_figure_widge**: create a figure capable of showing dynamic contents.
