import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Initial Raw Material Compositions
initial_composition_data = pd.DataFrame({
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

# Initial Target Composition Ranges
initial_target_data = pd.DataFrame({
    "Property": ["C", "Si", "Mn", "S", "P", "Cu", "Ni", "Mo", "Cr", "Hardness", "Tensile Strength"],
    "Min": [3.2, 1.5, 0.5, 0.01, 0.01, 0.05, 0.0, 0.0, 0.0, 180, 300],
    "Max": [3.6, 2.5, 1.0, 0.05, 0.02, 0.5, 0.2, 0.1, 0.1, 220, 350]
})

# Streamlit App for Charge Mix Optimization
def main():
    st.title("Charge Mix Optimization")

    st.write("### Editable Raw Material Compositions:")
    composition_data = st.data_editor(initial_composition_data, num_rows="dynamic")

    st.write("### Editable Target Composition Ranges:")
    target_data = st.data_editor(initial_target_data, num_rows="dynamic")

    # Get furnace size from user
    furnace_size = st.number_input("Enter Furnace Size (in tons):", min_value=0.1, step=0.1, value=10.0)

    # Select elements to include in optimization
    all_elements = list(initial_composition_data.columns[2:])
    selected_elements = st.multiselect("Select elements to include in optimization:", all_elements, default=all_elements)

    # Filter data based on selected elements
    composition_data = composition_data[["Material", "Cost"] + selected_elements]
    target_data = target_data[target_data["Property"].isin(selected_elements + ["Hardness", "Tensile Strength"])]

    # Define optimization parameters
    valid_materials = composition_data["Material"]
    costs = np.nan_to_num(composition_data["Cost"].values)  # Replace NaN with 0

    # Min and max constraints for each raw material
    bounds = [(0, furnace_size) for _ in valid_materials]  # Ensure total sum matches furnace size

    # Chemical and property constraints
    A_eq = np.array([[1] * len(valid_materials)])  # Sum of proportions = Furnace size
    b_eq = [furnace_size]

    # Build inequality constraints for chemical compositions and properties
    A_ub = []
    b_ub = []

    for i, row in target_data.iterrows():
        if row["Property"] in composition_data.columns:
            chem_coeffs = np.nan_to_num(composition_data[row["Property"]].values)  # Replace NaN with 0
            A_ub.append(-chem_coeffs)  # For Min constraints
            b_ub.append(-row["Min"] * furnace_size * 0.95)  # Scale by furnace size

            A_ub.append(chem_coeffs)  # For Max constraints
            b_ub.append(row["Max"] * furnace_size * 1.05)  # Scale by furnace size

    # Solve the optimization problem with feasibility adjustments
    res = linprog(
        c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs"
    )

    if res.success:
        optimized_proportions = res.x
        optimized_mix = pd.DataFrame({
            "Material": valid_materials,
            "Proportion (tons)": optimized_proportions,
            "Proportion (kg)": optimized_proportions * 1000  # Convert tons to kg
        })
        st.write("### Optimized Charge Mix:")
        st.dataframe(optimized_mix)
    else:
        st.warning("Optimization failed. Adjust constraints for feasibility.")
        st.write("Possible solutions:")
        st.write("- Relax min/max constraints by a small percentage.")
        st.write("- Increase the range for key elements.")
        st.write("- Allow higher flexibility in material proportions.")

if __name__ == "__main__":
    main()
