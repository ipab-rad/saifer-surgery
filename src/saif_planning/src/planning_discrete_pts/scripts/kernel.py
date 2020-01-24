from abc import ABCMeta, abstractmethod
from collections import namedtuple
import math

from funcsigs import signature
import warnings

import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform

#from ..metrics.pairwise import pairwise_kernels
#from ..base import clone
#from ..utils.validation import _num_samples
from sklearn.gaussian_process import *


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale


class Hyperparameter(namedtuple('Hyperparameter',
                                ('name', 'value_type', 'bounds',
                                 'n_elements', 'fixed'))):
    """A kernel hyperparameter's specification in form of a namedtuple.
    .. versionadded:: 0.18
    Attributes
    ----------
    name : string
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds
    value_type : string
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.
    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.
    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.
    fixed : bool, default: None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.
    """
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        if not isinstance(bounds, str) or bounds != "fixed":
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError("Bounds on %s should have either 1 or "
                                     "%d dimensions. Given are %d"
                                     % (name, n_elements, bounds.shape[0]))

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, n_elements, fixed)

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        return (self.name == other.name and
                self.value_type == other.value_type and
                np.all(self.bounds == other.bounds) and
                self.n_elements == other.n_elements and
                self.fixed == other.fixed)

class StationaryKernelMixin:
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).
    .. versionadded:: 0.18
    """

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return True


class NormalizedKernelMixin:
    """Mixin for kernels which are normalized: k(X, X)=1.
    .. versionadded:: 0.18
    """

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : sequence of length n_samples
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.ones(X.shape[0])

class Kernel(object):
    __metaclass__=ABCMeta
    """Base class for all kernels.
    .. versionadded:: 0.18
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if (parameter.kind != parameter.VAR_KEYWORD and
                    parameter.name != 'self'):
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError("scikit-learn kernels should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        for arg in args:
            try:
                value = getattr(self, arg)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            params[arg] = value
        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.
        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.
        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [getattr(self, attr) for attr in dir(self)
             if attr.startswith("hyperparameter_")]
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.
        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[i:i + hyperparameter.n_elements])
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.
        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [hyperparameter.bounds
                  for hyperparameter in self.hyperparameters
                  if not hyperparameter.fixed]
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map("{0:.3g}".format, self.theta)))

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""

    @abstractmethod
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : sequence of length n_samples
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

    @abstractmethod
    def is_stationary(self):
        """Returns whether the kernel is stationary. """

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on fixed-length feature
        vectors or generic objects. Defaults to True for backward
        compatibility."""
        return True

class RBF_Sep(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:
    k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, t_length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.t_length_scale = t_length_scale

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

########### MODIFICATION

# TODO separate lengthscales 

        X_1 = X[:,:-1]
        X_t = X[:, -1]
        Y_1 = Y[:,:-1]
        Y_t = Y[:, -1]

        if Y is None:
            dists1 = pdist(X_1 / length_scale, metric='sqeuclidean')
            K1 = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K1 = squareform(K1)
            np.fill_diagonal(K1, 1)

            dists2 = pdist(X_t / length_scale, metric='sqeuclidean')
            K2 = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K2 = squareform(K2)
            np.fill_diagonal(K2, 1)

            K = np.matmul(K1, K2)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists1 = cdist(X_1 / length_scale, Y_1 / length_scale,
                          metric='sqeuclidean')
            K1 = np.exp(-.5 * dists1)
            dists2 = cdist(X_t / length_scale, Y_t / length_scale,
                          metric='sqeuclidean')
            K2 = np.exp(-.5 * dists2)

            K = np.matmul(K1, K2)

##########

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])



if __name__ == "__main__":
    rbf = RBF_Sep(1.0)
