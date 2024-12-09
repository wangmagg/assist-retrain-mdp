import numpy as np
import torch
from src.actor import Actor, SwitchPDF

class Reward:
    """
    Reward model for the human-AI collaboration environment.

    Args:
        - tp_reward (float): reward for true positive
        - tn_reward (float): reward for true negative
        - type1_error_cost (float): cost for type 1 error
        - type2_error_cost (float): cost for type 2 error
        - retrain_cost (float): cost for retraining the AI model
    """

    def __init__(self, tp_reward, tn_reward, type1_error_cost, type2_error_cost, retrain_cost):
        self.tp_reward = tp_reward
        self.tn_reward = tn_reward
        self.type1_error_cost = type1_error_cost
        self.type2_error_cost = type2_error_cost
        self.retrain_cost = retrain_cost

    def __call__(self, action, decision, true_label):
        if isinstance(action, torch.Tensor):
            action = action.item()
        if action == 2:
            return self.retrain_cost
        if decision == 1 and true_label == 1:
            return self.tp_reward 
        elif decision == 0 and true_label == 0:
            return self.tn_reward 
        elif decision == 1 and true_label == 0:
            return self.type1_error_cost 
        elif decision == 0 and true_label == 1:
            return self.type2_error_cost 

class HumanAICollabEnv:
    """
    Environment for human-AI collaboration.

    Args:
        - human (Actor): human decision-maker
        - ai (Actor): AI decision-maker
        - switch_pdf (SwitchPDF): probability distribution for human switching to match AI
        - reward_mdl (Reward): reward model
        - human_update_config (dict): configuration for updating human model
        - ai_update_config (dict): configuration for updating AI model
        - switch_update_config (dict): configuration for updating switch model
    """
    def __init__(self, 
                 human: Actor, 
                 ai: Actor, 
                 switch_pdf: SwitchPDF,
                 reward_mdl: Reward, 
                 human_update_config,
                 ai_update_config,
                 switch_update_config,
                 seed=42):
        
        self.human = human
        self.ai = ai
        self.switch_pdf = switch_pdf
        self.reward_mdl = reward_mdl
        self.rng = np.random.default_rng(seed)

        self.human_update_config = human_update_config
        self.ai_update_config = ai_update_config
        self.switch_update_config = switch_update_config

        self.x_stream = None
        self.y_stream = None
        self.len_stream = None
        self.stream_idx = 0
        self.t = 0

    @property
    def n_obs(self):
        # Get the number of elements in the state vector
        n_human_params = len(self.human.decision_pdf.param_dict())
        n_ai_params = len(self.ai.decision_pdf.param_dict())
        n_switch_params = len(self.switch_pdf.param_dict())
        n_human_cm = self.human.cm.size
        n_ai_cm = self.ai.cm.size
        return 2 + n_human_params + n_ai_params + n_switch_params + n_human_cm + n_ai_cm
    
    @property
    def n_actions(self):
        # Get the number of actions
        return len(self.action_space())
    
    def set_dat_stream(self, dat_stream):
        # Set the data stream for the environment
        self.x_stream, self.y_stream = dat_stream
        self.len_stream = len(self.x_stream)

    def action_space(self):
        # 0: don't show AI recommendation
        # 1: show AI recommendation
        # 2: "retrain" the AI model then show AI recommendation
        return [0, 1, 2]

    def get_state(self):
        # Get the current state vector
        human_pdf_params = list(self.human.decision_pdf.param_dict().values())
        ai_pdf_params = list(self.ai.decision_pdf.param_dict().values())
        switch_params = list(self.switch_pdf.param_dict().values())
        human_cm = self.human.cm.flatten()
        ai_cm = self.ai.cm.flatten()
        x = np.atleast_1d(self.x_stream[self.stream_idx])
        t = np.atleast_1d(self.t)
        return np.concatenate([x, t, human_pdf_params, ai_pdf_params, switch_params, human_cm, ai_cm])

    def make_decision(self, x, y, action, ai_only=False):
        # Make a decision based on the covariate input
        human_decision = self.human.predict(x, y)
        ai_decision = self.ai.predict(x, y)

        # Use the AI-alone decision
        if ai_only:
            return ai_decision
        # Use the human-alone decision
        if action == 0:
            return human_decision
        # Use the human-AI decision
        else:
            ai_decision = self.ai.predict(x, y)
            p_switch = self.switch_pdf(x) 
            decision = self.rng.choice([ai_decision, human_decision], p=[p_switch, 1 - p_switch])
            return decision

    def get_reward(self, action, decision, y):
        return self.reward_mdl(action, decision, y)

    def _update_human(self):
        # update human to be more like AI
        if self.human_update_config is not None:
            ai_params = self.ai.decision_pdf.param_dict()
            for attr_name, val in self.human_update_config.items():
                self.human.decision_pdf.update_towards(
                    attr_name=attr_name,
                    target_val=ai_params[attr_name],
                    rng=self.rng,
                    **val
                )
    
    def _update_ai(self):
        # update AI performance
        if self.ai_update_config is not None:
            update_type = self.ai_update_config["update_type"]
            for attr_name, val in self.ai_update_config.items():
                if attr_name == "update_type":
                    continue
                self.ai.decision_pdf.update(attr_name=attr_name, rng=self.rng, update_type=update_type, **val)

    def _update_switch(self, reward):
        # update human's level of trust/reliance in AI
        update_type = self.switch_update_config["update_type"]
        for attr_name, val in self.switch_update_config.items():
            if attr_name == "update_type":
                continue
            self.switch_pdf.update(
                attr_name=attr_name,
                reward=reward,
                rng=self.rng,
                update_type=update_type,
                **val
            )
    
    def step(self, action, ai_only=False):
        # take a single step in the environment

        # if action is to retrain AI, reset the AI decision pdf
        if action == 2:
            self.ai.decision_pdf.reset()
            reward = self.get_reward(action, None, None)
        # otherwise, make a decision and update the AI decision pdf
        else:
            x = self.x_stream[self.stream_idx]
            y = self.y_stream[self.stream_idx]
            decision = self.make_decision(x, y, action, ai_only)
            reward = self.get_reward(action, decision, y)
            self._update_ai()
    
        # if action was to show AI recommendation, 
        # update the human decision performance and the switch pdf
        if action == 1:
            self._update_human()
            self._update_switch(reward)
        
        self.stream_idx += 1
        self.t += 1

        return reward

    def reset(self):
        # reset the environment to its initial state
        self.human.reset()
        self.ai.reset()
        self.switch_pdf.reset()
        self.stream_idx = 0
        self.t = 0
