import numpy as np
from scipy import stats

def update_confusion_matrix(cm, y_hat, y_true):
    if y_hat == 1 and y_true == 1:
        cm[1, 1] += 1
    elif y_hat == 0 and y_true == 0:
        cm[0, 0] += 1
    elif y_hat == 1 and y_true == 0:
        cm[0, 1] += 1
    elif y_hat == 0 and y_true == 1:
        cm[1, 0] += 1
    return cm

class DecisionPDF:
    """
    Conditional density function governing decision-making performance

    Args:
        - center (float): mean of the modified normal distribution
        - sigma (float): standard deviation of the modified normal distribution
        - min_p (float): minimum probability of correct decision
        - max_p_left (float): maximum probability of correct decision for x < center
        - max_p_right (float): maximum probability of correct decision for x >= center
    """
    def __init__(self, center, sigma, min_p, max_p_left, max_p_right):
        self.init_center = center
        self.init_sigma = sigma
        self.init_min_p = min_p
        self.init_max_p_left = max_p_left
        self.init_max_p_right = max_p_right

        self.center = center
        self.sigma = sigma
        self.min_p = min_p
        self.max_p_left = max_p_left
        self.max_p_right = max_p_right
    
    def __call__(self, x):
        alpha_l = (self.max_p_left - self.min_p) * np.sqrt(2 * np.pi) * self.sigma
        pdf_l = self.max_p_left - alpha_l * stats.norm.pdf(x, loc=self.center, scale=self.sigma)

        alpha_r = (self.max_p_right - self.min_p) * np.sqrt(2 * np.pi) * self.sigma
        pdf_r = self.max_p_right - alpha_r * stats.norm.pdf(x, loc=self.center, scale=self.sigma)

        pdf = pdf_l * (x < self.center) + pdf_r * (x >= self.center)

        return pdf
    

    def _update_linear(self, attr_name, p, beta, lb, ub, rng):
        # make a linear update to the specified attribute with probability p

        make_update = rng.binomial(1, p)
        if make_update:
            cur_val = getattr(self, attr_name)
            new_val = np.clip(cur_val + beta, lb, ub)
            setattr(self, attr_name, new_val)       

    def _update_geometric(self, attr_name, p, beta, lb, ub, rng):
        # make a multiplicative update to the specified attribute with probability p

        make_update = rng.binomial(1, p)
        if make_update:
            cur_val = getattr(self, attr_name)
            new_val = np.clip(cur_val * beta, lb, ub)
            setattr(self, attr_name, new_val)        
        
    def update(self, attr_name, rng, update_type='linear', **kwargs):
        # make an update to the specified attribute

        if update_type == 'linear':
            self._update_linear(attr_name=attr_name, rng=rng, **kwargs)
        elif update_type == 'geometric':
            self._update_geometric(attr_name=attr_name, rng=rng, **kwargs)

    def update_towards(self, target_val, attr_name, p, beta, rng):
        # make an update to the specified attribute towards the target value with probability p

        make_update = rng.binomial(1, p)
        if make_update:
            cur_val = getattr(self, attr_name)
            diff = target_val - cur_val
            new_val = cur_val + diff * np.abs(beta)
            setattr(self, attr_name, new_val)

    def reset(self):
        # reset the decision pdf to its initial state
        self.center = self.init_center
        self.sigma = self.init_sigma
        self.min_p = self.init_min_p
        self.max_p_left = self.init_max_p_left
        self.max_p_right = self.init_max_p_right

    def init_param_dict(self):
        return {
            'center': self.init_center,
            'sigma': self.init_sigma,
            'min_p': self.init_min_p,
            'max_p_left': self.init_max_p_left,
            'max_p_right': self.init_max_p_right
        }
    
    def param_dict(self):
        return {
            'center': self.center,
            'sigma': self.sigma,
            'min_p': self.min_p,
            'max_p_left': self.max_p_left,
            'max_p_right': self.max_p_right
        }
    
class SwitchPDF:
    """
    Conditional density function governing probability that
    the human "switches" their decision to match the AI's 

    Args:
        - center (float): mean of the modified normal distribution
        - sigma (float): standard deviation of the modified normal distribution
        - min_p (float): minimum probability of switching
        - max_p (float): maximum probability of switching
    """
    def __init__(self, center, sigma, min_p, max_p):

        self.init_center = center
        self.init_sigma = sigma
        self.init_min_p = min_p
        self.init_max_p = max_p

        self.center = center
        self.sigma = sigma
        self.min_p = min_p
        self.max_p = max_p
    
    def __call__(self, x):
        alpha = (self.max_p - self.min_p) * np.sqrt(2 * np.pi) * self.sigma
        return self.min_p + alpha * stats.norm.pdf(x, loc=self.center, scale=self.sigma)
    
    def _update_linear(self, attr_name, reward, p, beta, lb, ub, rng):
        # Make a linear update to the specified attribute with probability p
        if reward <= 0:
            # If the reward is negative, the probability of a positive udpate 
            # is min(p, 1 - p)
            is_pos = rng.binomial(1, min(p, 1 - p))
        else:
            # If the reward is positive, the probability of a positive update
            # is max(p, 1 - p)
            is_pos = rng.binomial(1, max(p, 1 - p))
        cur_val = getattr(self, attr_name)
        if is_pos:
            new_val = np.clip(cur_val + np.abs(beta), lb, ub)
        else:
            new_val = np.clip(cur_val - np.abs(beta), lb, ub)
        setattr(self, attr_name, new_val)

    def _update_geometric(self, attr_name, reward, p, beta, lb, ub, rng):
        # Make a multiplicative update to the specified attribute with probability p
        if reward <= 0:
            is_pos = rng.binomial(1, min(p, 1 - p))
        else:
            is_pos = rng.binomial(1, max(p, 1 - p))
        cur_val = getattr(self, attr_name)
        if is_pos:
            new_val = np.clip(cur_val * np.abs(beta), lb, ub)
        else:
            new_val = np.clip(cur_val / np.abs(beta), lb, ub)
        setattr(self, attr_name, new_val)

    def update(self, attr_name, reward, rng, update_type='linear', **kwargs):
        if update_type == 'linear':
            self._update_linear(attr_name=attr_name, rng=rng, reward=reward, **kwargs)
        elif update_type == 'geometric':
            self._update_geometric(attr_name=attr_name, rng=rng, reward=reward, **kwargs)

    def reset(self):
        self.center = self.init_center
        self.sigma = self.init_sigma
        self.min_p = self.init_min_p
        self.max_p = self.init_max_p

    def init_param_dict(self):
        return {
            'center': self.init_center,
            'sigma': self.init_sigma,
            'min_p': self.init_min_p,
            'max_p': self.init_max_p
        }
    
    def param_dict(self):
        return {
            'center': self.center,
            'sigma': self.sigma,
            'min_p': self.min_p,
            'max_p': self.max_p
        }

    
class Actor:
    """
    Human or AI decision-maker

    Args:
        - decision_pdf (DecisionPDF): conditional density function governing decision-making performance
        - init_cm (np.ndarray): initial confusion matrix
        - seed (int): random seed
    """
    def __init__(self, decision_pdf: DecisionPDF, init_cm=None, seed=42):
        self.decision_pdf = decision_pdf
        self.rng = np.random.default_rng(seed)

        if init_cm is None:
            self.init_cm = np.zeros((2, 2))
        self.cm = np.copy(self.init_cm)
        self.mdl = None

    def update_cm(self, y_hat, y_true):
        # Update the confusion matrix based on the executed decision 
        # and the ground truth correct decision
        self.cm = update_confusion_matrix(self.cm, y_hat, y_true)

    def predict(self, x, y_true):
        # Make a decision based on the covariate input
        # Probability of being correct is determined by the decision pdf
        p_correct = self.decision_pdf(x)
        y_pred = self.rng.choice([y_true, 1 - y_true], p=[p_correct, 1 - p_correct])
        self.update_cm(y_pred, y_true)
        
        return y_pred
    
    def reset(self):
        # Reset the confusion matrix to its initial state
        self.cm = np.copy(self.init_cm)
        self.decision_pdf.reset()