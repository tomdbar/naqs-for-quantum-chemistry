import numpy as np
import torch
from enum import Enum

def real(x):
    return x[..., 0]

def imag(x):
    return x[..., 1]

def exp(x):
    amp, ph = real(x).exp(), imag(x)
    return torch.stack([amp * ph.cos(),
                        amp * ph.sin()],
                        -1)

class VarType(Enum):
    TORCH = 0
    NUMPY = 1

def is_complex(x):
    if (x.dim() > 1) and x.shape[-1]:
        return True
    else:
        return False

def to_complex(x, y=None):
    var_type = VarType.TORCH if torch.is_tensor(x) else VarType.NUMPY
    if y is None:
        if var_type is VarType.TORCH:
            y = torch.zeros_like(x)
        else:
            y = np.zeros_like(x)

    if var_type is VarType.TORCH:
        z = torch.stack([x, y], -1)
    else:
        z = torch.FloatTensor(np.stack([x, y], -1))

    return z

def matmul(x, y):
    y = y.to(x)
    re = torch.matmul(real(x), real(y)).sub_(torch.matmul(imag(x), imag(y)))
    im = torch.matmul(real(x), imag(y)).add_(torch.matmul(imag(x), real(y)))

    return to_complex(re, im)

def scalar_mult(x, y):
    """A function that computes the product between complex matrices and scalars,
    complex vectors and scalars or two complex scalars.
    """
    y = y.to(x)

    re = real(x) * real(y) - imag(x) * imag(y)
    im = real(x) * imag(y) + imag(x) * real(y)

    return to_complex(re, im)


def inner_prod(x, y):
    """A function that returns the inner product of two complex vectors,
    x and y (<x|y>).
    :param x: A complex vector.
    :type x: torch.Tensor
    :param y: A complex vector.
    :type y: torch.Tensor
    :raises ValueError: If x and y are not complex vectors with their first
                        dimensions being 2, then the function will not execute.
    :returns: The inner product, :math:`\\langle x\\vert y\\rangle`.
    :rtype: torch.Tensor
    """
    y = y.to(x)

    if x.dim() == 2 and y.dim() == 2:
        return to_complex(
            torch.dot(real(x), real(y)) + torch.dot(imag(x), imag(y)),
            torch.dot(real(x), imag(y)) - torch.dot(imag(x), real(y)),
        )
    elif x.dim() == 1 and y.dim() == 1:
        return to_complex(
            (real(x) * real(y)) + (imag(x) * imag(y)),
            (real(x) * imag(y)) - (imag(x) * real(y)),
        )
    else:
        raise ValueError("Unsupported input shapes!")

def outer_prod(x, y):
    """A function that returns the outer product of two complex vectors, x
    and y.
    """
    if x.dim() != y.dim():
        raise ValueError("The input dimensions don't match.")

    if x.dim() == 2:
        x = x.unsqueeze(0)  # (N, 2) --> (m_batch=1, N, 2)
        y = y.unsqueeze(0)  # (N, 2) --> (m_batch=1, N, 2)

    if x.dim() != 3:
        raise ValueError("An input is not of the right dimension :", x.shape)

    mX_batch, nX, dX = x.shape
    mY_batch, nY, dY = y.shape

    if dX != 2 or dY != 2:
        raise ValueError("Must pass complex tensors.")
    if mX_batch != mY_batch:
        raise ValueError("Batch sizes must match")

    z = torch.zeros(mX_batch, nX, nY, 2, dtype=torch.double, device=x.device)

    batch_ger = lambda a, b: torch.einsum('bi,bj->bij', (a, b))

    z[...,0] = batch_ger(x[...,0], y[...,0]) - batch_ger(x[...,1], y[...,1])
    z[...,1] = batch_ger(x[...,0], y[...,1]) + batch_ger(x[...,1], y[...,0])

    return z


def conj(x):
    """Returns the element-wise complex conjugate of the argument.
    :param x: A complex tensor.
    :type x: torch.Tensor
    :returns: The complex conjugate of x.
    :rtype: torch.Tensor
    """
    return to_complex(real(x), -imag(x))

def absolute_value(x):
    """Returns the complex absolute value elementwise.
    :param x: A complex tensor.
    :type x: torch.Tensor
    :returns: A real tensor.
    :rtype: torch.Tensor
    """
    x_star = conj(x)
    return real(scalar_mult(x, x_star)).sqrt_()

def np_to_torch(x):
    return torch.FloatTensor(np.stack([np.real(x), np.imag(x)], -1))

def torch_to_numpy(x):
    return np.array(real(x).numpy()+1j*imag(x).numpy())

def from_polar(amps, phase):
    return to_complex(amps*phase.cos(), amps*phase.sin())

def rect2polar(x):
    amp = x.pow(2).sum(-1).pow(0.5)
    phase = (x[...,0] / amp).acos()
    return torch.to_complex([amp, phase], -1)

def polar2rect(x):
    return to_complex([x[...,0] * x[...,1].cos(),
                       x[...,0] * x[...,1].sin()])