import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# -------------------------------------------
# Example Coefficients (Replace with real data)
# -------------------------------------------
HARDNESS_COEFFS = {
    "constant": 50,
    "C": 40,
    "Si": -15,
    "Mn": 10,
    "S": -5,
    "P": -5,
    "Cu": 4,
    "Ni": 3,
    "Mo": 2,
    "Cr": 3
}

TENSILE_COEFFS = {
    "constant": 300,
    "C": 20,
    "Si": -10,
    "Mn": 25,
    "S": -10,
    "P": -2,
    "Cu": 5,
    "Ni": 8,
    "Mo": 5,
    "Cr": 5
}

# -------------------------------------------
# Initial Data
# -------------------------------------------
initial_composition_data = pd.DataFrame({
    "Material": [
        "HMS", "Shredded Scrap", "Pig Iron", "Ferro-Silicon", "Ferro-Manganese", "FeSiMg", "Copper Scrap", "FeMo", "Carburiser"
    ],
    "Cost": [34, 32, 42, 120, 150, 210, 600, 2200, 80],
    "C": [2.5, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
    "Si": [0.5, 0.7, 1.0, 75.0, 0.0, 50.0, 0.0, 0.0, 0.0],
    "Mn": [0.3, 0.5, 0.1, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "S": [0.05, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    "P": [0.02, 0.015, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Cu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0],
    "Ni": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0, 0.0],
    "Mo": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
    "Cr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]
})

initial_target_data = pd.DataFrame({
    "Property": ["C", "Si", "Mn", "S", "P", "Cu", "Ni", "Mo", "Cr", "Hardness", "Tensile Strength"],
    "Min": [3.2, 1.5, 0.5, 0.01, 0.01, 0.05, 0.0, 0.0, 0.0, 180, 300],
    "Max": [3.6, 2.5, 1.0, 0.05, 0.02, 0.5, 0.2, 0.1, 0.1, 220, 350]
})

def main():
    st.title("Charge Mix Optimization")

    # Section 1: Composition Data
    st.write("### Editable Raw Material Compositions:")
    composition_data = st.data_editor(initial_composition_data, num_rows="dynamic")

    # Section 2: Target Data
    st.write("### Editable Target Composition Ranges:")
    target_data = st.data_editor(initial_target_data, num_rows="dynamic")

    # Section 3: Furnace Size
    furnace_size = st.number_input("Enter Furnace Size (in tons):", min_value=0.1, step=0.1, value=10.0)

    # Section 4: Per-Material Bounds
    bounds = []
    st.sidebar.write("### Set Material Constraints")
    for material in composition_data["Material"]:
        min_val = st.sidebar.slider(f"Min proportion for {material} (tons)", 0.0, furnace_size, 0.0)
        max_val = st.sidebar.slider(f"Max proportion for {material} (tons)", 0.0, furnace_size, furnace_size)
        bounds.append((min_val, max_val))

    # Section 5: Select elements to include in optimization
    all_elements = list(initial_composition_data.columns[2:])
    selected_elements = st.multiselect(
        "Select elements to include in optimization:",
        all_elements,
        default=all_elements
    )

    # Filter data based on selected elements
    composition_data = composition_data[["Material", "Cost"] + selected_elements]
    # Keep only relevant rows in target_data
    target_data = target_data[target_data["Property"].isin(selected_elements + ["Hardness", "Tensile Strength"])]

    # --------------------------------------------------------------------------------
    # Build Hardness & Tensile Coefficients for each Raw Material
    # --------------------------------------------------------------------------------
    def get_hardness_coefficient(row):
        """Compute the 'per-ton' hardness contributed by this raw material."""
        # Sum up k_element * row[element] for each element
        hardness_val = HARDNESS_COEFFS["constant"]  # Start from the constant
        for elem in selected_elements:
            # If that element has a coefficient, multiply by content
            if elem in HARDNESS_COEFFS:
                hardness_val += HARDNESS_COEFFS[elem] * row[elem]
        return hardness_val

    def get_tensile_coefficient(row):
        """Compute the 'per-ton' tensile contributed by this raw material."""
        tensile_val = TENSILE_COEFFS["constant"]
        for elem in selected_elements:
            if elem in TENSILE_COEFFS:
                tensile_val += TENSILE_COEFFS[elem] * row[elem]
        return tensile_val

    # Compute an array of hardness/tensile values for each row (raw material)
    hardness_array = composition_data.apply(get_hardness_coefficient, axis=1).values
    tensile_array = composition_data.apply(get_tensile_coefficient, axis=1).values

    # Next, we want the *total* hardness to be between Hardness Min and Max.
    # The *total* hardness = sum( hardness_array[i] * x[i] ) / sum(x[i])?
    # Or do we treat the "constant" as a purely additive term?
    #
    # Often, the simpler approach is to use the "per-ton" formula but
    # skip the global 'constant' in the raw material portion. Instead,
    # we incorporate the constant after summation. That means we’d have:
    #
    #   total_hardness = CONSTANT + Σ( coefficient[i] * x[i] )
    #
    # For the example’s sake, let's proceed with the linear approach below
    # where we treat hardness_array[i] as the entire contribution from
    # material i (including the constant). Because x[i] is in tons,
    # the sum will yield an absolute hardness that is scaled by tons,
    # which might or might not be what you want physically.
    #
    # If you want an *average* hardness, you’d have to divide the sum
    # by the total mass. For demonstration, we’ll keep it simple:
    #
    #  total_hardness = sum( hardness_array[i]* x[i] ), and that must
    #  lie between Hardness_min and Hardness_max.

    # Identify the required min/max from target_data
    hardness_min = target_data.loc[target_data["Property"] == "Hardness", "Min"].values[0]
    hardness_max = target_data.loc[target_data["Property"] == "Hardness", "Max"].values[0]
    tensile_min = target_data.loc[target_data["Property"] == "Tensile Strength", "Min"].values[0]
    tensile_max = target_data.loc[target_data["Property"] == "Tensile Strength", "Max"].values[0]

    # We'll set up inequalities for hardness and tensile:
    #
    # hardness_min <= SUM(hardness_array[i]*x[i]) <= hardness_max
    #  => - SUM(hardness_array[i]*x[i]) <= - hardness_min
    #  => SUM(hardness_array[i]*x[i])   <= hardness_max
    #
    # tensile_min <= SUM(tensile_array[i]*x[i]) <= tensile_max

    A_ub = []
    b_ub = []

    # Hardness >= hardness_min
    # => -hardness_array . x <= -hardness_min
    A_ub.append(-hardness_array)
    b_ub.append(-hardness_min)

    # Hardness <= hardness_max
    A_ub.append(hardness_array)
    b_ub.append(hardness_max)

    # Tensile >= tensile_min
    A_ub.append(-tensile_array)
    b_ub.append(-tensile_min)

    # Tensile <= tensile_max
    A_ub.append(tensile_array)
    b_ub.append(tensile_max)

    # Next, we also have composition constraints for each element
    # if you want each element to be within a certain min/max in the *final melt*.
    # For example:
    #    sum( C_i * x[i] ) / sum(x[i]) >= C_min
    #    sum( C_i * x[i] ) / sum(x[i]) <= C_max
    #
    # or if you want it absolute:
    #    sum( C_i * x[i] ) >= some_val
    #
    # The code snippet below shows how you might handle the "average" composition constraints:

    for prop in selected_elements:
        # Get min, max from target_data if they exist
        row_target = target_data[target_data["Property"] == prop]
        if not row_target.empty:
            min_val = row_target["Min"].values[0]
            max_val = row_target["Max"].values[0]

            # We want: min_val <= (Σ (prop_i * x[i])) / total_tons <= max_val
            # => min_val * total_tons <= Σ (prop_i * x[i]) <= max_val * total_tons
            #
            # Because sum(x[i]) = furnace_size (b_eq below),
            # we can rewrite:
            #    min_val * furnace_size <= Σ (prop_i * x[i]) <= max_val * furnace_size

            prop_array = composition_data[prop].values

            # Lower bound
            A_ub.append(-prop_array)  # - Σ(prop_i * x[i]) <= - (min_val * furnace_size)
            b_ub.append(-min_val * furnace_size)

            # Upper bound
            A_ub.append(prop_array)   # Σ(prop_i * x[i]) <= max_val * furnace_size
            b_ub.append(max_val * furnace_size)

    # Convert A_ub, b_ub to arrays
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Now define equality constraint to ensure total mass = furnace_size
    A_eq = np.array([[1]*len(composition_data)])
    b_eq = [furnace_size]

    # Solve the LP
    c = np.nan_to_num(composition_data["Cost"].values)  # objective: min cost

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    # ----------------------------------------------------------
    # Report Results
    # ----------------------------------------------------------
    if res.success:
        optimized_proportions = res.x
        optimized_mix = pd.DataFrame({
            "Material": composition_data["Material"],
            "Proportion (tons)": optimized_proportions,
            "Proportion (kg)": optimized_proportions * 1000
        })

        st.write("### Optimized Charge Mix:")
        st.dataframe(optimized_mix)

        # Calculate final composition for each selected element
        final_comp = {}
        for elem in selected_elements:
            final_comp[elem] = np.dot(optimized_proportions, composition_data[elem].values)

        # If you want average composition per ton:
        #   final_comp[elem] / furnace_size
        # This is optional, depending on whether you want absolute or average.

        st.write("### Final Composition (Absolute):")
        st.json(final_comp)

        # Calculate total Hardness and Tensile
        total_hardness = np.dot(hardness_array, optimized_proportions)
        total_tensile = np.dot(tensile_array, optimized_proportions)

        st.write(f"#### Final Hardness: {total_hardness:.2f}")
        st.write(f"#### Final Tensile Strength: {total_tensile:.2f}")

    else:
        st.warning("Optimization failed. Adjust constraints for feasibility.")

if __name__ == "__main__":
    main()
