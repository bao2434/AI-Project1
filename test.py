# from Agents import *
# import time

# if __name__ == "__main__":

#     # Use your sample observation here.
    
#     observation = {

#         # Board configuration
#         "board_size": 9,

#         # Player information
#         "player_id": 0,
#         "current_player": 0,

#         # Game progress
#         "turn": 10,
#         "max_turns": 200,
#         "done": False,
#         "winner": None,

#         # Player states
#         "you": {
#             "position": [4, 4],
#             "goal_row": 8,
#             "walls_remaining": 10
#         },

#         "opponent": {
#             "position": [2, 4],
#             "goal_row": 0,
#             "walls_remaining": 10
#         },

#         # Placed walls
#         "walls": {
#             "horizontal": [],
#             "vertical": []
#         },

#         # Shortest-path distances to your goal row
#         "distances": [
#             [8, 8, 8, 8, 8, 8, 8, 8, 8],
#             [7, 7, 7, 7, 7, 7, 7, 7, 7],
#             [6, 6, 6, 6, 6, 6, 6, 6, 6],
#             [5, 5, 5, 5, 5, 5, 5, 5, 5],
#             [4, 4, 4, 4, 4, 4, 4, 4, 4],
#             [3, 3, 3, 3, 3, 3, 3, 3, 3],
#             [2, 2, 2, 2, 2, 2, 2, 2, 2],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         ],

#         # Legal player movements
#         "legal_moves": [
#             [3, 4],
#             [4, 3],
#             [4, 5],
#             [5, 4]
#         ],

#         # Legal wall placements
#         "legal_walls": []
#     }


#     # --------------------------------------------------------
#     # Benchmark prediction
#     # --------------------------------------------------------

#     num_runs = 100
#     total_time = 0.0

#     for _ in range(num_runs):

#         start_time = time.perf_counter()

#         model = WrongWayBot()

#         action = model(observation)

#         end_time = time.perf_counter()

#         elapsed = end_time - start_time
#         total_time += elapsed

#     # --------------------------------------------------------
#     # Results
#     # --------------------------------------------------------

#     average_time = total_time / num_runs

#     print("\n========== PPO Prediction Benchmark ==========")

#     print("Predicted action:", action)
#     print(f"Number of runs: {num_runs}")
#     print(f"Total time: {total_time:.6f} seconds")
#     print(f"Average prediction: {average_time * 1000:.3f} ms")
#     print(f"Average prediction: {average_time:.6f} seconds")
#     print(f"Predictions per second: {1 / average_time:.2f}")

#     print("==============================================")


