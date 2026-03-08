from flask import Flask, jsonify, request, send_from_directory
from env import Game2048Env
import os
import numpy as np
import torch
from agent import DQNAgent

app = Flask(__name__, static_folder='static')
env = Game2048Env()

# Initialize AI Agent
ai_agent = DQNAgent()
try:
    ai_agent.q_network.load_state_dict(torch.load('best_model.pth'))
    print("Loaded best_model.pth successfully for AI play.")
except Exception as e:
    print(f"Could not load best_model.pth: {e}")
env = Game2048Env()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    info = {
        'board': env.board.tolist(),
        'score': 0, # Depending on how we track score, env.py tracks reward but not cumulative score. We might just use max tile or calculate it.
        'game_over': env.is_game_over()
    }
    return jsonify(info)

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    action = data.get('action') # 0: Up, 1: Down, 2: Left, 3: Right
    
    if action not in [0, 1, 2, 3]:
        return jsonify({'error': 'Invalid action'}), 400
        
    next_state, reward, done, info = env.step(action)
    
    response = {
        'board': env.board.tolist(),
        'reward': reward,
        'game_over': done,
        'valid_move': info.get('valid_move', True)
    }
    return jsonify(response)

@app.route('/api/reset', methods=['POST'])
def reset_game():
    env.reset()
    info = {
        'board': env.board.tolist(),
        'game_over': env.is_game_over()
    }
    return jsonify(info)

@app.route('/api/ai_move', methods=['POST'])
def make_ai_move():
    if env.is_game_over():
        return jsonify({'error': 'Game is already over', 'game_over': True}), 400
        
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return jsonify({'error': 'No valid actions', 'game_over': True}), 400
        
    board_state = np.ascontiguousarray(env.board)
    action = ai_agent.select_action(board_state, epsilon=0.0, valid_actions=valid_actions)
    next_state, reward, done, info = env.step(action)
    
    response = {
        'board': env.board.tolist(),
        'reward': reward,
        'game_over': done,
        'valid_move': True,
        'action': action
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
