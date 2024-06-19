from mbfit.polynomials import FragmentedPolynomialGenerator
from mbfit.fitting.fragmented import generate_fit_code
from mbfit import generate_poly_input

# This file can be completely empty, but its required for legacy reasons. Will be removed at some point.
settings_path = "scratches/[FOLDER]/settings.ini"
poly_in_path = "scratches/[FOLDER]/[NAME].in"
poly_log_path = "scratches/[FOLDER]/[NAME].log"

generate_poly_input(settings_path, "[NAME]", poly_in_path)

with open(poly_in_path, "a") as f:
    f.write("""
set_name['[NAME]']
[ADD_ATOMS]
""")

polynomial_generator = FragmentedPolynomialGenerator(settings_path)

polynomial_generator.add_polynomial(
    poly_in_path, 3, poly_log_path,
    monomer_indices=[MONOMER_INDICES],
    nmer_indices=[NMER_INDICES],
) # DOUBLE-CHECK monomer_indices and nmer_indices!!!

polynomial_generator.generate_cpp_file("scratches/[FOLDER]/[NAME].cpp")

generate_fit_code("scratches/[FOLDER]/[NAME].cpp", "scratches/[FOLDER]/[NAME]_fit_code", True)