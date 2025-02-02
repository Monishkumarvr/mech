import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------
# 1) Hardness/Tensile Coefficients (Excluding the Constant)
# --------------------------------------------------------------------
HARDNESS_CONSTANT = 50.0  # Base hardness
HARDNESS_COEFFS = {
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

TENSILE_CONSTANT = 300.0  # Base tensile
TENSILE_COEFFS = {
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

# --------------------------------------------------------------------
# 2) Sample Raw Materials
# --------------------------------------------------------------------
initial_composition_data = pd.DataFrame({
    "Material": [
        "HMS", "Shredded Scrap", "Pig Iron", "Ferro-Silicon", 
        "Ferro-Manganese", "FeSiMg", "Copper Scrap", "FeMo", 
        "Carburiser"
    ],
    "Cost": [34, 32, 42, 120, 150, 210, 600, 2200, 80],
    "C":  [2.5, 3.0, 4.0,   0.0,  0.0,  0.0,  0.0,   0.0,  90.0],
    "Si": [0.5, 0.7, 1.0,  75.0,  0.0, 50.0,  0.0,   0.0,   0.0],
    "Mn": [0.3, 0.5, 0.1,   0.0, 70.0,  0.0,  0.0,   0.0,   0.0],
    "S":  [0.05,0.03,0.01,  0.0,  0.0,  0.0,  0.0,   0.0,   0.1],
    "P":  [0.02,0.015,0.01, 0.0,  0.0,  0.0,  0.0,   0.0,   0.0],
    "Cu": [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 90.0,   0.0,   0.0],
    "Ni": [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  70.0,   0.0],
    "Mo": [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  10.0,   0.0],
    "Cr": [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   5.0,   0.0],
})

# --------------------------------------------------------------------
# 3) Sample Target Ranges
# --------------------------------------------------------------------
initial_target_data = pd.DataFrame({
    "Property": [
        "C", "Si", "Mn", "S", "P", "Cu", "Ni", "Mo", "Cr", 
        "Hardness", "Tensile Strength"
    ],
    "Min": [3.2, 1.5, 0.5, 0.01, 0.01, 0.05, 0.0, 0.0, 0.0, 
            180, 300],
    "Max": [3.6, 2.5, 1.0, 0.05, 0.02, 0.5,  0.2, 0.1, 0.1, 
            220, 350]
})

def main():
    st.title("Charge Mix Optimization")

    # -----------------------------------------------
    # A) Let users edit composition and target data
    # -----------------------------------------------
    st.subheader("Raw Material Compositions (editable):")
    composition_data = st.data_editor(
        initial_composition_data, num_rows="dynamic"
    )

    st.subheader("Target Composition Ranges (editable):")
    target_data = st.data_editor(
        initial_target_data, num_rows="dynamic"
    )

    # -----------------------------------------------
    # B) Furnace size
    # -----------------------------------------------
    furnace_size = st.number_input(
        "Enter Furnace Size (tons):", 
        min_value=0.1, step=0.1, value=10.0
    )

    # -----------------------------------------------
    # C) Bounds for each raw material
    # -----------------------------------------------
    st.sidebar.header("Material Proportion Constraints")
    bounds = []
    for material in composition_data["Material"]:
        min_val = st.sidebar.slider(
            f"Min {material} (tons)", 0.0, furnace_size, 0.0
        )
        max_val = st.sidebar.slider(
            f"Max {material} (tons)", 0.0, furnace_size, furnace_size
        )
        bounds.append((min_val, max_val))

    # -----------------------------------------------
    # D) Select which elements to use
    # -----------------------------------------------
    all_elements = list(composition_data.columns[2:])  # e.g. C, Si, Mn, ...
    selected_elements = st.multiselect(
        "Select elements to include in optimization:",
        all_elements,
        default=all_elements
    )

    # Filter composition_data columns
    composition_data = composition_data[["Material", "Cost"] + selected_elements]

    # Filter target_data rows
    target_data = target_data[
        target_data["Property"].isin(selected_elements + ["Hardness", "Tensile Strength"])
    ]

    # -----------------------------------------------
    # E) Build Hardness & Tensile arrays (exclude constant)
    # -----------------------------------------------
    def hardness_contrib_per_ton(row):
        val = 0.0
        for elem, coeff in HARDNESS_COEFFS.items():
            if elem in row.index:
                val += coeff * row[elem]
        return val

    def tensile_contrib_per_ton(row):
        val = 0.0
        for elem, coeff in TENSILE_COEFFS.items():
            if elem in row.index:
                val += coeff * row[elem]
        return val

    hardness_array = composition_data.apply(hardness_contrib_per_ton, axis=1).values
    tensile_array = composition_data.apply(tensile_contrib_per_ton, axis=1).values

    # -----------------------------------------------
    # F) Extract Hardness/Tensile min/max
    # -----------------------------------------------
    hardness_min = target_data.loc[target_data["Property"] == "Hardness", "Min"].values[0]
    hardness_max = target_data.loc[target_data["Property"] == "Hardness", "Max"].values[0]
    tensile_min = target_data.loc[target_data["Property"] == "Tensile Strength", "Min"].values[0]
    tensile_max = target_data.loc[target_data["Property"] == "Tensile Strength", "Max"].values[0]

    # --------------------------------------------------------
    # G) Build inequality constraints (A_ub x <= b_ub)
    #    for Hardness and Tensile (average-based)
    # --------------------------------------------------------
    A_ub = []
    b_ub = []

    # Hardness (avg) = HARDNESS_CONSTANT + (1/furnace_size)*sum(hardness_array[i]*x[i])
    # => hardness_min <= ... <= hardness_max
    # => hardness_min - HARDNESS_CONSTANT <= sum(...)/furnace_size <= hardness_max - HARDNESS_CONSTANT
    # => furnace_size*(hardness_min - HARDNESS_CONSTANT) <= sum(...) <= furnace_size*(hardness_max - HARDNESS_CONSTANT)

    # Hardness >= hardness_min
    A_ub.append(-hardness_array)  # -sum(...) <= -lower_bound
    b_ub.append(
        -furnace_size*(hardness_min - HARDNESS_CONSTANT)
    )

    # Hardness <= hardness_max
    A_ub.append(hardness_array)
    b_ub.append(
        furnace_size*(hardness_max - HARDNESS_CONSTANT)
    )

    # Tensile (avg) = TENSILE_CONSTANT + (1/furnace_size)*sum(tensile_array[i]*x[i])
    # => tensile_min <= ... <= tensile_max
    # => furnace_size*(tensile_min - TENSILE_CONSTANT) <= sum(...) <= furnace_size*(tensile_max - TENSILE_CONSTANT)

    # Tensile >= tensile_min
    A_ub.append(-tensile_array)
    b_ub.append(
        -furnace_size*(tensile_min - TENSILE_CONSTANT)
    )

    # Tensile <= tensile_max
    A_ub.append(tensile_array)
    b_ub.append(
        furnace_size*(tensile_max - TENSILE_CONSTANT)
    )

    # ----------------------------------------------------
    # H) Composition constraints for each selected element
    # ----------------------------------------------------
    # If target_data says "C" min=3.2, max=3.6, interpret that as
    #   3.2 <= average_C <= 3.6
    # => 3.2 * furnace_size <= sum(C_i * x[i]) <= 3.6 * furnace_size

    for prop in selected_elements:
        row_target = target_data[target_data["Property"] == prop]
        if not row_target.empty:
            min_val = row_target["Min"].values[0]
            max_val = row_target["Max"].values[0]

            prop_array = composition_data[prop].values

            # Lower bound
            A_ub.append(-prop_array)
            b_ub.append(-furnace_size * min_val)

            # Upper bound
            A_ub.append(prop_array)
            b_ub.append(furnace_size * max_val)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    # ----------------------------------------------------
    # I) Equality constraint for total mass
    # ----------------------------------------------------
    # sum(x[i]) == furnace_size
    A_eq = np.array([[1]*len(composition_data)], dtype=float)
    b_eq = [furnace_size]

    # ----------------------------------------------------
    # J) Objective: Minimize Cost
    # ----------------------------------------------------
    c = np.nan_to_num(composition_data["Cost"].values, nan=0.0)

    # ----------------------------------------------------
    # K) Solve with linprog
    # ----------------------------------------------------
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    # ----------------------------------------------------
    # L) Report solution
    # ----------------------------------------------------
    if res.success:
        st.success("Optimization Succeeded!")
        optimized_proportions = res.x

        # Summarize
        solution_df = pd.DataFrame({
            "Material": composition_data["Material"],
            "Proportion (tons)": optimized_proportions,
            "Proportion (kg)": optimized_proportions * 1000
        })
        st.write("### Optimized Charge Mix")
        st.dataframe(solution_df)

        # Final composition (absolute)
        final_composition = {}
        for elem in selected_elements:
            final_composition[elem] = np.dot(optimized_proportions, composition_data[elem].values)

        st.write("### Final Composition (Absolute Amount)")
        st.json(final_composition)

        # Average composition per ton:
        avg_composition = {k: v/furnace_size for k,v in final_composition.items()}
        st.write("### Final Composition (Average per ton)")
        st.json(avg_composition)

        # Compute final average Hardness & Tensile
        total_hardness_contrib = np.dot(hardness_array, optimized_proportions)
        avg_hardness = HARDNESS_CONSTANT + (total_hardness_contrib / furnace_size)

        total_tensile_contrib = np.dot(tensile_array, optimized_proportions)
        avg_tensile = TENSILE_CONSTANT + (total_tensile_contrib / furnace_size)

        st.write(f"**Final Average Hardness:** {avg_hardness:.2f}")
        st.write(f"**Final Average Tensile Strength:** {avg_tensile:.2f}")

    else:
        st.error("Optimization failed. Try relaxing constraints or checking data consistency.")
        st.write("Solver Message:", res.message)


if __name__ == "__main__":
    main()
