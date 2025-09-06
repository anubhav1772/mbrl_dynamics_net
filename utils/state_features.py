class StateFeatures:
    def __init__(self):
        self.feature_slices = {
            "gravity_vector": slice(0, 3),
            "x_vel": slice(3, 4),
            "y_vel": slice(4, 5),
            "yaw_vel": slice(5, 6),
            "body_height": slice(6, 7),
            "step_freq": slice(7, 8),
            "gait": slice(8, 11),
            "durations": slice(11, 12),
            "footswing_height": slice(12, 13),
            "body_pitch": slice(13, 14),
            "body_roll": slice(14, 15),
            "stance_width": slice(15, 16),
            "stance_length": slice(16, 17),
            "aux_reward": slice(17, 18),
            "dof_pos": slice(18, 30),
            "dof_vel": slice(30, 42),
            "actions": slice(42, 54),
            "clock_inputs": slice(54, 58),
        }

    def __getitem__(self, key):
        """Allow dictionary-style access, e.g., features["dof_pos"]"""
        return self.feature_slices[key]

    def __getattr__(self, key):
        """Allow attribute-style access, e.g., features.dof_pos"""
        if key in self.feature_slices:
            return self.feature_slices[key]
        raise AttributeError(f"'StateFeatures' object has no attribute '{key}'")

    def all(self):
        """Return the full feature mapping dictionary"""
        return self.feature_slices


if __name__ == '__main__':
    features = StateFeatures()

    # Dictionary style
    print(features["dof_pos"])   # slice(18, 30)

    # Attribute style
    print(features.dof_vel)      # slice(30, 42)

    # Get indices for slicing tensors
    import torch
    state = torch.randn(5, 58)   # batch of 5 states
    dof_pos = state[:, features.dof_pos]   # shape (5, 12)
    dof_vel = state[:, features.dof_vel]   # shape (5, 12)

    # Iterate over all features
    for name, sl in features.all().items():
        print(name, state[:, sl].shape)

