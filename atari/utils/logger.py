import pandas as pd


class CSVLogger:
    def __init__(self, log_path, tensorboard_writer=None):
        self.log_path = log_path
        self.writer = tensorboard_writer
        self.results = pd.DataFrame(columns=[
            "global_step", "episode", "return", "SNR", "EmaSNR", "AdamPlusSNR",
            "m", "v", "g", "pwr_noise", "runtime", "lr"
        ])

    def log_episode(self, step, episode, info_dict):
        df = pd.DataFrame([{
            "global_step": step,
            "episode": episode,
            **info_dict
        }])
        self.results = pd.concat([self.results, df], ignore_index=True)
        self.results.to_csv(self.log_path, index=False)

        # Optional tensorboard logging
        if self.writer:
            if "return" in info_dict:
                self.writer.add_scalar("charts/episodic_return", info_dict["return"], step)
            if "lr" in info_dict:
                self.writer.add_scalar("charts/learning_rate", info_dict["lr"], step)
            # Add more as needed
