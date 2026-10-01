from fractions import Fraction
import math

import sympy as sp


REGISTRY = {}


KIND = {}


def kernel(op_id):
    def deco(f):
        REGISTRY[op_id] = f
        KIND[op_id] = "kernel"
        return f
    return deco


def atom(atom_id):
    def deco(f):
        REGISTRY[atom_id] = f
        KIND[atom_id] = "knowledge"
        f.atom_id = atom_id
        return f
    return deco


def render(v):
    if isinstance(v, tuple):
        return "(" + ", ".join(render(x) for x in v) + ")"
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, sp.Interval):
        lo = "(" if v.left_open else "["
        hi = ")" if v.right_open else "]"
        return f"{lo}{render(v.start)}, {render(v.end)}{hi}"
    if isinstance(v, sp.FiniteSet):
        return "{" + ", ".join(render(x) for x in sorted(v, key=str)) + "}"
    if isinstance(v, sp.MatrixBase):
        return "[" + "; ".join(", ".join(render(x) for x in v.row(i))
                               for i in range(v.rows)) + "]"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


@kernel("kernel.add")
def _add(x, y): return x + y


@kernel("kernel.subtract")
def _subtract(x, y): return x - y


@kernel("kernel.multiply")
def _multiply(x, y): return x * y


@kernel("kernel.divide")
def _divide(x, y):
    if y == 0:
        raise ValueError("kernel.divide by zero")
    return Fraction(x, y) if isinstance(x, int) and isinstance(y, int) else x / y


@kernel("kernel.negate")
def _negate(x): return -x


@kernel("kernel.integer_power")
def _integer_power(x, n): return x ** n


@kernel("kernel.equal")
def _equal(x, y): return x == y


@kernel("kernel.less_than")
def _less_than(x, y): return x < y


@kernel("kernel.make_point")
def _make_point(x, y): return (x, y)


@kernel("kernel.make_vector")
def _make_vector(xs): return tuple(xs)


@kernel("kernel.get_coordinate")
def _get_coordinate(p, i): return p[i]


@kernel("kernel.get_entry")
def _get_entry(v, i): return v[i]


def poly(pairs, v="x"):
    terms = {}
    for p, k in pairs:
        terms[p] = terms.get(p, 0) + k
    out = ""
    for p in sorted((p for p, k in terms.items() if k), reverse=True):
        k = terms[p]
        body = "" if p == 0 else v if p == 1 else f"{v}^{p}"
        mag = "" if abs(k) == 1 and body else str(abs(k))
        piece = f"{mag}{body}"
        if not out:
            out = f"−{piece}" if k < 0 else piece
        else:
            out += f" {'−' if k < 0 else '+'} {piece}"
    return out or "0"


def bracket(pairs, v="x"):
    return "(" + poly(pairs, v) + ")"


def quad(a, b, c, v="x"):
    return poly([(2, a), (1, b), (0, c)], v)


def cubic(a, b, c, d, v="x"):
    return poly([(3, a), (2, b), (1, c), (0, d)], v)


def lineq(m, x1, y1):
    m = Fraction(m)
    b = Fraction(y1) - m * Fraction(x1)
    if m == 0:
        return f"y = {pf(b.numerator, b.denominator)}"
    ms = "x" if m == 1 else "-x" if m == -1 else f"{pf(m.numerator, m.denominator)}x"
    if b == 0:
        return f"y = {ms}"
    sign = "+" if b > 0 else "−"
    return f"y = {ms} {sign} {pf(abs(b.numerator), b.denominator)}"


def pf(num, den):
    f = Fraction(num, den)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def coeff_poly(coeffs):
    top = len(coeffs) - 1
    return poly([(top - i, k) for i, k in enumerate(coeffs)])


def display(v, mode=None):
    if mode == "polynomial":
        return coeff_poly(v)
    if mode == "capitalize":
        return str(v).capitalize()
    if mode and "{}" in mode:
        return mode.format(render(v))
    return render(v)


@kernel("kernel.greater_than")
def _greater_than(x, y): return x > y


@kernel("kernel.less_equal")
def _less_equal(x, y): return x <= y


@kernel("kernel.greater_equal")
def _greater_equal(x, y): return x >= y


@kernel("kernel.not_equal")
def _not_equal(x, y): return x != y


@kernel("kernel.logical_and")
def _logical_and(x, y): return bool(x) and bool(y)


@kernel("kernel.logical_or")
def _logical_or(x, y): return bool(x) or bool(y)


@kernel("kernel.logical_not")
def _logical_not(x): return not bool(x)


@kernel("kernel.absolute_value")
def _absolute_value(x):
    if isinstance(x, sp.Expr) and not x.is_real:
        raise ValueError("absolute_value is for real numbers; use a modulus atom")
    return abs(x)


@kernel("kernel.make_set")
def _make_set(xs): return sp.FiniteSet(*xs)


@kernel("kernel.make_interval")
def _make_interval(a, b, left_open=False, right_open=False):
    if a > b:
        raise ValueError("interval endpoints out of order")
    return sp.Interval(a, b, bool(left_open), bool(right_open))


@kernel("kernel.make_complex")
def _make_complex(re_part, im_part): return sp.sympify(re_part) + sp.sympify(im_part) * sp.I


@kernel("kernel.real_part")
def _real_part(z): return sp.re(sp.sympify(z))


@kernel("kernel.imaginary_part")
def _imaginary_part(z): return sp.im(sp.sympify(z))


@kernel("kernel.make_matrix")
def _make_matrix(rows): return sp.Matrix([list(r) for r in rows])


@kernel("kernel.get_row")
def _get_row(m, i): return sp.Matrix(m).row(i)


@kernel("kernel.get_column")
def _get_column(m, j): return sp.Matrix(m).col(j)
