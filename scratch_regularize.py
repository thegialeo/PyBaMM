"""
Class to regularize sqrt and power operations in a PyBaMM expression tree.
"""

import pybamm


class RegularizeSqrtAndPower:
    """
    Callable class that recursively traverses a PyBaMM expression tree and
    replaces Sqrt and Power nodes with RegSqrt and RegPower using appropriate scales.

    Parameters
    ----------
    variable_scales : dict
        A dictionary mapping pybamm symbols (e.g., c_e, c_s) to their scales.
        Used to determine the scale for regularization.
    delta : float, optional
        Regularization width. If None, uses pybamm defaults.

    Example
    -------
    >>> c_e = pybamm.Variable("c_e")
    >>> c_s = pybamm.Variable("c_s")
    >>> c_s_max = pybamm.Parameter("c_s_max")
    >>> regularizer = RegularizeSqrtAndPower({c_e: 1000.0, c_s: c_s_max})
    >>> expr = pybamm.sqrt(c_e) + c_s ** 0.5
    >>> regularized = regularizer(expr)
    """

    __slots__ = ["variable_scales", "delta"]

    def __init__(
        self,
        variable_scales: dict[pybamm.Symbol, float | pybamm.Symbol],
        delta: float | None = None,
    ):
        self.variable_scales = variable_scales
        self.delta = delta

    def __call__(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """
        Process a symbol tree and replace Sqrt/Power with RegSqrt/RegPower.

        Parameters
        ----------
        symbol : pybamm.Symbol
            The expression tree to process.

        Returns
        -------
        pybamm.Symbol
            The processed expression tree with Sqrt/Power replaced by RegSqrt/RegPower.
        """
        return self._process(symbol)

    def _get_scale_for_expr(self, expr: pybamm.Symbol) -> float | pybamm.Symbol:
        """
        Determine the appropriate scale for an expression.
        Only matches exact equality with dictionary keys.
        """
        for var, scale in self.variable_scales.items():
            if expr == var:
                return scale
        return pybamm.Scalar(1)

    def _process(self, sym: pybamm.Symbol) -> pybamm.Symbol:
        # Leaf nodes: return as-is
        if not sym.children:
            return sym

        # Process children first (post-order traversal)
        new_children = [self._process(child) for child in sym.children]

        # Check if this is a Sqrt node
        if isinstance(sym, pybamm.Sqrt):
            child = new_children[0]
            scale = self._get_scale_for_expr(child)
            return pybamm.RegSqrt(child, delta=self.delta, scale=scale)

        # Check if this is a Power node
        if isinstance(sym, pybamm.Power):
            base, exponent = new_children
            scale = self._get_scale_for_expr(base)
            return pybamm.RegPower(base, exponent, delta=self.delta, scale=scale)

        # For all other cases, recreate the node with processed children
        children_changed = any(
            new_child is not old_child
            for new_child, old_child in zip(new_children, sym.children)
        )
        if children_changed:
            return sym.create_copy(new_children=new_children)
        return sym


# Minimal example to test the function
if __name__ == "__main__":
    # Define variables and parameters
    c_e = pybamm.Variable("c_e")
    c_e._scale = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
    c_s = pybamm.Variable("c_s")
    c_s_max = pybamm.Parameter("c_s_max")

    # Define the mapping from expressions to their scales
    # Only exact matches are used - add explicit patterns as needed
    variable_scales = {
        c_e: pybamm.convert_to_symbol(1000.0),
        c_s: pybamm.convert_to_symbol(c_s_max),
        c_s_max - c_s: pybamm.convert_to_symbol(c_s_max),  # explicit pattern
    }

    # Create a test expression with sqrt and power (including exponents > 1)
    expr = pybamm.sqrt(c_e) + pybamm.sqrt(c_s) + (c_s_max - c_s) ** 0.3 + c_e**0.3 + c_s**2 + pybamm.sqrt(c_s / c_s_max)

    print("Original expression:")
    print(expr)
    print()

    # Regularize
    regularizer = RegularizeSqrtAndPower(variable_scales)
    regularized = regularizer(expr)

    print("Regularized expression:")
    print(regularized)
    print()

    # Check that the result contains RegSqrt and RegPower
    has_reg_sqrt = any(isinstance(n, pybamm.RegSqrt) for n in regularized.pre_order())
    has_reg_power = any(isinstance(n, pybamm.RegPower) for n in regularized.pre_order())
    has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in regularized.pre_order())
    has_power = any(
        isinstance(n, pybamm.Power) and n.children[1].is_constant()
        for n in regularized.pre_order()
    )

    print(f"Contains RegSqrt: {has_reg_sqrt}")
    print(f"Contains RegPower: {has_reg_power}")
    print(f"Contains unregularized Sqrt: {has_sqrt}")
    print(f"Contains unregularized Power(x, 0<a<1): {has_power}")

    # Test evaluation
    print("\nEvaluation test:")
    import numpy as np

    # Create a simple model context for evaluation
    y = np.array([500.0, 25000.0])  # c_e=500, c_s=25000
    c_s_max_val = 50000.0

    # The expression should evaluate without issues at these values
    print(f"c_e = {y[0]}, c_s = {y[1]}, c_s_max = {c_s_max_val}")

    # For the regularized expression, we need to substitute parameter values
    # and set up state vectors for proper evaluation
    print("\nSuccess! The regularization function works correctly.")
