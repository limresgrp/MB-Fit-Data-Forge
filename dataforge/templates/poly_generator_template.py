from mbfit.polynomials import FragmentedPolynomialGenerator
from mbfit.fitting.fragmented import generate_fit_code
from mbfit import generate_poly_input

# This file can be completely empty, but its required for legacy reasons. Will be removed at some point.
settings_path = "scratches/settings.ini"
poly_in_path = "scratches/[NAME].in"
poly_log_path = "scratches/[NAME].log"

generate_poly_input(settings_path, "[NAME]", poly_in_path)

with open(poly_in_path, "a") as f:
    f.write("""
set_name['[NAME]']
[ADD_ATOMS]
""")

polynomial_generator = FragmentedPolynomialGenerator(settings_path)

polynomial_generator.add_polynomial(poly_in_path, 4, poly_log_path)

polynomial_generator.generate_cpp_file("scratches/[NAME].cpp")

generate_fit_code("scratches/[NAME].cpp", "scratches/[NAME]_fit_code", True)