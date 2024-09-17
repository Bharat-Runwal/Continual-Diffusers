# class ReplayScenario:
#     def __init__(self, args):
#         '''
#         args : dict - the arguments to pass to the replay
#         '''
#         self.replay_type = args.replay_type
#         # based on the replay_type instantiate the replay
#         if self.replay_type == 'none':
#             self.replay = None
#         elif self.replay_type == 'reservoir':
#             self.replay = BufferReplay(args)
#         elif self.replay_type == 'generative':
#             self.replay = GenerativeReplay(args)
#         elif self.replay_type == 'exemplar':
#             self.replay = ExemplarReplay(args)
#         elif self.replay_type == 'hybrid':
#             self.replay = HybridReplay(args)
#         else:
#             raise ValueError(f"Replay type {self.replay_type} not recognized.")

#     def __call__(self, scenario):
#         return scenario(self.replay)
