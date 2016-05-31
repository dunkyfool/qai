import tensorflow as tf
import numpy as np

def rnn():
  lstm = rnn_cell.BasicLSTMCell(lstm_size)
  # Initial state of the LSTM memory.
  state = tf.zeros([batch_size, lstm.state_size])

  loss = 0.0
  for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)
  pass


def rnn_step():
  words = tf.placeholder(tf.int32, [batch_size, num_steps])

  lstm = rnn_cell.BasicLSTMCell(lstm_size)
  # Initial state of the LSTM memory.
  initial_state = state = tf.zeros([batch_size, lstm.state_size])

  for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

  final_state = state


  numpy_state = initial_state.eval()
  total_loss = 0.0
  for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss

def mul_rnn():
  lstm = rnn_cell.BasicLSTMCell(lstm_size)
  stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

  initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
  for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = stacked_lstm(words[:, i], state)

    # The rest of the code.
    # ...

  final_state = state


if__name__=='__main__':
  pass
