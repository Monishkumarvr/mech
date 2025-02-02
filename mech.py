import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Hardcoded Raw Material Compositions
composition_data = pd.DataFrame({
    "Material": [
        "HMS", "Shredded Scrap", "Pig Iron", "Ferro-Silicon", "Ferro-Manganese", "FeSiMg", "Copper Scrap", "FeMo", "Carburiser"
    ],
    "Cost": [300, 320, 400, 2000, 1800, 2500, 450, 1500, 800],
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

# Hardcoded Target Composition Ranges
target_data = pd.DataFrame({
    "Property": ["C", "Si", "Mn", "S", "P", "Cu", "Ni", "Mo", "Cr", "Hardness", "Tensile Strength"],
    "Min": [3.2, 1.5, 0.5, 0.01, 0.01, 0.05, 0.0, 0.0, 0.0, 180, 300],
    "Max": [3.6, 2.5, 1.0, 0.05, 0.02, 0.5, 0.2, 0.1, 0.1, 220, 350]
})

# Streamlit App for Charge Mix Optimization
def main():
    st.title("Charge Mix Optimization")

    st.write("### Raw Material Compositions:")
    st.dataframe(composition_data)

    st.write("### Target Composition Ranges:")
    st.dataframe(target_data)

    # Define optimization parameters
    valid_materials = composition_data["Material"]
    costs = composition_data["Cost"].values

    # Min and max constraints for each raw material
    bounds = []
    st.sidebar.write("### Set Material Constraints")
    for material in valid_materials:
        min_val = st.sidebar.slider(f"Min proportion for {material}", 0.0, 1.0, 0.05)
        max_val = st.sidebar.slider(f"Max proportion for {material}", 0.0, 1.0, 0.5)
        bounds.append((min_val, max_val))

    # Chemical and property constraints
    A_eq = np.array([[1] * len(valid_materials)])  # Sum of proportions = 1
    b_eq = [1]

    # Build inequality constraints for chemical compositions and properties
    A_ub = []
    b_ub = []

    for i, row in target_data.iterrows():
        if row["Property"] in composition_data.columns:
            chem_coeffs = composition_data[row["Property"]].values
            A_ub.append(-chem_coeffs)  # For Min constraints
            b_ub.append(-row["Min"])

            A_ub.append(chem_coeffs)  # For Max constraints
            b_ub.append(row["Max"])

    # Solve the optimization problem
    res = linprog(
        c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs"
    )

    if res.success:
        optimized_proportions = res.x
        optimized_mix = pd.DataFrame({
            "Material": valid_materials,
            "Proportion": optimized_proportions,
        })

        # Calculate resulting properties
        resulting_properties = {}
        for prop in target_data["Property"]:
            if prop in composition_data.columns:
                resulting_properties[prop] = np.dot(optimized_proportions, composition_data[prop].values)

        resulting_properties["Hardness"] = 200 + 50 * resulting_properties.get("C", 0) - 10 * resulting_properties.get("Si", 0)
        resulting_properties["Tensile Strength"] = 300 + 30 * resulting_properties.get("Mn", 0) - 5 * resulting_properties.get("Si", 0)

        st.write("### Optimized Charge Mix:")
        st.dataframe(optimized_mix)

        st.write("### Resulting Properties:")
        st.json(resulting_properties)
    else:
        st.error(f"Optimization failed: {res.message}")

if __name__ == "__main__":
    main()
