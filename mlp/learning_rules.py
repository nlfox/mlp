# -*- coding: utf-8 -*-
"""Learning rules.

This module contains classes implementing gradient based learning rules.
"""

import numpy as np


class GradientDescentLearningRule(object):
    """Simple (stochastic) gradient descent learning rule.

    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form

        p[i] := p[i] - learning_rate * dE/dp[i]

    With `learning_rate` a positive scaling parameter.

    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, learning_rate=1e-3):
        """Creates a new learning rule object.

        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.

        """
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = learning_rate

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        This must be called before `update_params` is first called.

        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
        self.params = params

    def reset(self):
        """Resets any additional state variables to their intial values.

        For this learning rule there are no additional state variables so we
        do nothing here.
        """
        pass

    def update_params(self, grads_wrt_params):
        """Applies a single gradient descent update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        for param, grad in zip(self.params, grads_wrt_params):
            param -= self.learning_rate * grad


class AdamLearningRule(GradientDescentLearningRule):
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(AdamLearningRule, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.moms_1 = []
        self.moms_2 = []
        self.step_count = 0

    def initialise(self, params):
        super(AdamLearningRule, self).initialise(params)
        self.moms_1 = []
        for param in self.params:
            self.moms_1.append(np.zeros_like(param))
        self.moms_2 = []
        for param in self.params:
            self.moms_2.append(np.zeros_like(param))
        self.step_count = 0

    def reset(self):
        for mom_1, mom_2 in zip(self.moms_1, self.moms_2):
            mom_1 *= 0
            mom_2 *= 0

    def update_params(self, grads_wrt_params):
        for param, mom_1, mom_2, grad in zip(
                self.params, self.moms_1, self.moms_2, grads_wrt_params):
            mom_1 *= self.beta_1
            mom_1 += (1.0 - self.beta_1) * grad
            mom_2 *= self.beta_2
            mom_2 += (1.0 - self.beta_2) * grad ** 2
            alpha_t = self.learning_rate * (1.0 - self.beta_2 ** (self.step_count + 1)) ** 0.5 / (
                1.0 - self.beta_1 ** (self.step_count + 1))
            param -= alpha_t * mom_1 / (mom_2 ** 0.5 + self.epsilon)
        self.step_count += 1


class AdaGradLearningRule(GradientDescentLearningRule):
    """Adaptive gradients (AdaGrad) learning rule.
    First-order gradient-descent based learning rule which normalises gradient
    updates by a running sum of the past squared gradients.
    References:
      [1]: Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization. Duchi, Haxan and Singer, 2011
    """

    def __init__(self, learning_rate=1e-2, epsilon=1e-8):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
            epsilon: 'Softening' parameter to stop updates diverging when
                sums of squared gradients are close to zero. Should be set to
                a small positive value.
        """
        super(AdaGradLearningRule, self).__init__(learning_rate)
        self.epsilon = epsilon

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.
        This must be called before `update_params` is first called.
        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
        super(AdaGradLearningRule, self).initialise(params)
        self.sum_sq_grads = []
        for param in self.params:
            self.sum_sq_grads.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their initial values.
        For this learning rule this corresponds to zeroing all the sum of
        squared gradient states.
        """
        for sum_sq_grad in self.sum_sq_grads:
            sum_sq_grad *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        for param, sum_sq_grad, grad in zip(
                self.params, self.sum_sq_grads, grads_wrt_params):
            sum_sq_grad += grad ** 2
            param -= (self.learning_rate * grad /
                      (sum_sq_grad + self.epsilon) ** 0.5)


class RMSPropLearningRule(GradientDescentLearningRule):
    def __init__(self, learning_rate=1e-3, beta=0.9, epsilon=1e-8):
        super(RMSPropLearningRule, self).__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.moms = []

    def initialise(self, params):
        super(RMSPropLearningRule, self).initialise(params)
        self.moms = []
        for param in self.params:
            self.moms.append(np.zeros_like(param))

    def reset(self):
        for mom in self.moms:
            mom *= 0.

    def update_params(self, grads_wrt_params):
        for param, mom, grad in zip(
                self.params, self.moms, grads_wrt_params):
            mom *= self.beta
            mom += (1. - self.beta) * grad ** 2
            param -= self.learning_rate * grad / (mom + self.epsilon) ** 0.5
