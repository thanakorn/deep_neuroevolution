from ray.experimental.queue import Queue

class EvaluationLogger:
    def __init__(self, get_log_data, log_size=100000):
        self.log = Queue()
        self.get_log_data = get_log_data
        
    def log_data(self, env):
        self.log.put_nowait(self.get_log_data(env))