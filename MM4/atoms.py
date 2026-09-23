from fractions import Fraction
import sympy as sp

from kernel import atom

# Topic 1. The logarithmic function

@atom("func.log.definition")
def log_definition(base, value):
    # ACMMM151
    # a^x = b  <=>  x = log_a(b)
    base = sp.sympify(base)
    value = sp.sympify(value)

    if base.is_positive is not True or base == 1:
        raise ValueError("logarithm base must be positive and different from 1")
    if value.is_positive is not True:
        raise ValueError("logarithm argument must be positive")

    return sp.simplify(sp.log(value) / sp.log(base))


@atom("func.log.product_rule")
def log_product_rule(base, x, y):
    # ACMMM152
    # log_a(xy) = log_a(x) + log_a(y)
    base = sp.sympify(base)
    x = sp.sympify(x)
    y = sp.sympify(y)

    if base.is_positive is not True or base == 1:
        raise ValueError("logarithm base must be positive and different from 1")
    if x.is_positive is not True or y.is_positive is not True:
        raise ValueError("logarithm arguments must be positive")

    return sp.simplify(sp.log(x, base) + sp.log(y, base))


@atom("func.log.quotient_rule")
def log_quotient_rule(base, x, y):
    # ACMMM152
    # log_a(x/y) = log_a(x) - log_a(y)
    base = sp.sympify(base)
    x = sp.sympify(x)
    y = sp.sympify(y)

    if base.is_positive is not True or base == 1:
        raise ValueError("logarithm base must be positive and different from 1")
    if x.is_positive is not True or y.is_positive is not True:
        raise ValueError("logarithm arguments must be positive")

    return sp.simplify(sp.log(x, base) - sp.log(y, base))


@atom("func.log.power_rule")
def log_power_rule(base, x, power):
    # ACMMM152
    # log_a(x^k) = k log_a(x)
    base = sp.sympify(base)
    x = sp.sympify(x)
    power = sp.sympify(power)

    if base.is_positive is not True or base == 1:
        raise ValueError("logarithm base must be positive and different from 1")
    if x.is_positive is not True:
        raise ValueError("logarithm argument must be positive")

    return sp.simplify(power * sp.log(x, base))


@atom("func.log.decibel")
def log_decibel(intensity, reference):
    # ACMMM154
    # Decibel scale: L = 10 log_10(I / I_0)
    intensity = sp.sympify(intensity)
    reference = sp.sympify(reference)

    if intensity.is_positive is not True or reference.is_positive is not True:
        raise ValueError("intensity and reference must be positive")

    return sp.simplify(10 * sp.log(intensity / reference, 10))


@atom("func.log.equation")
def log_equation(base, value):
    # ACMMM157
    # Solve log_a(x) = value, so x = a^value.
    base = sp.sympify(base)
    value = sp.sympify(value)

    if base.is_positive is not True or base == 1:
        raise ValueError("logarithm base must be positive and different from 1")

    return sp.simplify(base ** value)


@atom("func.log.natural")
def log_natural(value):
    # ACMMM159
    # ln(x) = log_e(x)
    value = sp.sympify(value)

    if value.is_positive is not True:
        raise ValueError("natural logarithm argument must be positive")

    return sp.log(value)


@atom("func.log.exp_ln_inverse")
def log_exp_ln_inverse(value):
    # ACMMM160
    # Uses the inverse relationship e^(ln x) = x.
    value = sp.sympify(value)

    if value.is_positive is not True:
        raise ValueError("value must be positive")

    return sp.simplify(sp.exp(sp.log(value)))


@atom("func.log.ln_derivative")
def log_ln_derivative(value):
    # ACMMM161
    # d/dx (ln x) = 1/x, evaluated at x = value.
    value = sp.sympify(value)

    if value.is_zero is True:
        raise ValueError("value must not be zero")

    return sp.simplify(1 / value)


# Topic 2. Continuous random variables and the normal distribution

@atom("prob.continuous.relative_frequency")
def continuous_relative_frequency(count, total):
    # ACMMM164
    # Estimate a probability from a relative frequency: count / total.
    if not isinstance(count, int) or not isinstance(total, int):
        raise TypeError("count and total must be integers")
    if total <= 0:
        raise ValueError("total must be positive")
    if count < 0 or count > total:
        raise ValueError("count must satisfy 0 <= count <= total")

    return Fraction(count, total)


@atom("prob.continuous.probability")
def continuous_probability(pdf, variable, lower, upper):
    # ACMMM165
    # P(lower < X < upper) = integral of f(x) over the interval.
    variable = sp.sympify(variable)
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    pdf = sp.sympify(pdf)

    if not bool(lower < upper):
        raise ValueError("lower must be less than upper")

    return sp.simplify(sp.integrate(pdf, (variable, lower, upper)))


@atom("prob.continuous.cdf")
def continuous_cdf(pdf, variable, lower_support, value):
    # ACMMM165
    # F(x) = integral of f(t) from the lower support endpoint to x.
    variable = sp.sympify(variable)
    lower_support = sp.sympify(lower_support)
    value = sp.sympify(value)
    pdf = sp.sympify(pdf)

    if not bool(lower_support <= value):
        raise ValueError("value must not be below the lower support")

    return sp.simplify(
        sp.integrate(pdf, (variable, lower_support, value))
    )


@atom("prob.continuous.expected_value")
def continuous_expected_value(pdf, variable, lower, upper):
    # ACMMM166
    # E(X) = integral of x f(x) dx.
    variable = sp.sympify(variable)
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    pdf = sp.sympify(pdf)

    if not bool(lower < upper):
        raise ValueError("lower must be less than upper")

    return sp.simplify(
        sp.integrate(variable * pdf, (variable, lower, upper))
    )


@atom("prob.continuous.variance")
def continuous_variance(pdf, variable, lower, upper):
    # ACMMM166
    # Var(X) = E(X^2) - [E(X)]^2.
    variable = sp.sympify(variable)
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    pdf = sp.sympify(pdf)

    if not bool(lower < upper):
        raise ValueError("lower must be less than upper")

    mean = sp.integrate(variable * pdf, (variable, lower, upper))
    second_moment = sp.integrate(
        variable ** 2 * pdf, (variable, lower, upper)
    )

    return sp.simplify(second_moment - mean ** 2)


@atom("prob.continuous.linear_transform")
def continuous_linear_transform(mean, standard_deviation, multiplier, shift):
    # ACMMM167
    # If Y = aX + b:
    # E(Y) = a E(X) + b and SD(Y) = |a| SD(X).
    mean = sp.sympify(mean)
    standard_deviation = sp.sympify(standard_deviation)
    multiplier = sp.sympify(multiplier)
    shift = sp.sympify(shift)

    if standard_deviation.is_nonnegative is not True:
        raise ValueError("standard deviation must be non-negative")

    return (
        sp.simplify(multiplier * mean + shift),
        sp.simplify(sp.Abs(multiplier) * standard_deviation),
    )


@atom("prob.normal.standardize")
def normal_standardize(value, mean, standard_deviation):
    # ACMMM169
    # z = (x - mu) / sigma.
    value = sp.sympify(value)
    mean = sp.sympify(mean)
    standard_deviation = sp.sympify(standard_deviation)

    if standard_deviation.is_positive is not True:
        raise ValueError("standard deviation must be positive")

    return sp.simplify(
        (value - mean) / standard_deviation
    )

# Topic 3. Interval estimates for proportions

@atom("prob.proportion.distribution")
def proportion_distribution(p, sample_size):
    # ACMMM174
    # E(p_hat) = p and SD(p_hat) = sqrt[p(1-p)/n].
    p = sp.sympify(p)
    sample_size = sp.sympify(sample_size)

    if p.is_real is not True:
        raise ValueError("p must be real")
    if not bool(0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    if not bool(sample_size > 0):
        raise ValueError("sample size must be positive")

    return (
        sp.simplify(p),
        sp.simplify(sp.sqrt(p * (1 - p) / sample_size)),
    )


@atom("prob.proportion.confidence_interval")
def proportion_confidence_interval(sample_proportion, sample_size, z):
    # ACMMM178
    # p_hat +/- z sqrt[p_hat(1-p_hat)/n].
    sample_proportion = sp.sympify(sample_proportion)
    sample_size = sp.sympify(sample_size)
    z = sp.sympify(z)

    if not bool(0 <= sample_proportion <= 1):
        raise ValueError("sample proportion must be between 0 and 1")
    if not bool(sample_size > 0):
        raise ValueError("sample size must be positive")
    if not bool(z >= 0):
        raise ValueError("z must be non-negative")

    margin = z * sp.sqrt(
        sample_proportion * (1 - sample_proportion) / sample_size
    )

    return (
        sp.simplify(sample_proportion - margin),
        sp.simplify(sample_proportion + margin),
    )


@atom("prob.proportion.margin_error")
def proportion_margin_error(sample_proportion, sample_size, z):
    # ACMMM179
    # E = z sqrt[p_hat(1-p_hat)/n].
    sample_proportion = sp.sympify(sample_proportion)
    sample_size = sp.sympify(sample_size)
    z = sp.sympify(z)

    if not bool(0 <= sample_proportion <= 1):
        raise ValueError("sample proportion must be between 0 and 1")
    if not bool(sample_size > 0):
        raise ValueError("sample size must be positive")
    if not bool(z >= 0):
        raise ValueError("z must be non-negative")

    return sp.simplify(
        z * sp.sqrt(
            sample_proportion * (1 - sample_proportion) / sample_size
        )
    )


@atom("prob.proportion.standardize")
def proportion_standardize(sample_proportion, population_proportion, sample_size):
    # ACMMM176
    # (p_hat - p) / sqrt[p_hat(1-p_hat)/n].
    sample_proportion = sp.sympify(sample_proportion)
    population_proportion = sp.sympify(population_proportion)
    sample_size = sp.sympify(sample_size)

    if not bool(0 <= sample_proportion <= 1):
        raise ValueError("sample proportion must be between 0 and 1")
    if not bool(0 <= population_proportion <= 1):
        raise ValueError("population proportion must be between 0 and 1")
    if not bool(sample_size > 0):
        raise ValueError("sample size must be positive")

    denominator = sp.sqrt(
        sample_proportion * (1 - sample_proportion) / sample_size
    )
    if denominator == 0:
        raise ValueError("standardising denominator must be non-zero")

    return sp.simplify(
        (sample_proportion - population_proportion) / denominator
    )
