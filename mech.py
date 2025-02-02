import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Initial Raw Material Compositions
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

    # Min and max constraints for each raw material
    bounds = []
    st.sidebar.write("### Set Material Constraints")
    for material in composition_data["Material"]:
        min_val = st.sidebar.slider(f"Min proportion for {material} (tons)", 0.0, furnace_size, 0.0)
        max_val = st.sidebar.slider(f"Max proportion for {material} (tons)", 0.0, furnace_size, furnace_size)
        bounds.append((min_val, max_val))

    # Select elements to include in optimization
    all_elements = list(initial_composition_data.columns[2:])
    selected_elements = st.multiselect("Select elements to include in optimization:", all_elements, default=all_elements)

    # Filter data based on selected elements
    composition_data = composition_data[["Material", "Cost"] + selected_elements]
    target_data = target_data[target_data["Property"].isin(selected_elements + ["Hardness", "Tensile Strength"])]

    # Solve the optimization problem
    res = linprog(
        c=np.nan_to_num(composition_data["Cost"].values), A_eq=np.array([[1] * len(composition_data)]),
        b_eq=[furnace_size], A_ub=np.array([]), b_ub=np.array([]), bounds=bounds, method="highs"
    )

    if res.success:
        optimized_proportions = res.x
        optimized_mix = pd.DataFrame({
            "Material": composition_data["Material"],
            "Proportion (tons)": optimized_proportions,
            "Proportion (kg)": optimized_proportions * 1000
        })
        
        # Compute final hardness and tensile strength
        final_C = np.dot(optimized_proportions, composition_data.get("C", 0).values)
        final_Si = np.dot(optimized_proportions, composition_data.get("Si", 0).values)
        final_Mn = np.dot(optimized_proportions, composition_data.get("Mn", 0).values)
        
        final_hardness = 200 + 50 * final_C - 10 * final_Si
        final_tensile = 300 + 30 * final_Mn - 5 * final_Si
        
        st.write("### Optimized Charge Mix:")
        st.dataframe(optimized_mix)
        
        st.write(f"### Final Hardness: {final_hardness:.2f}")
        st.write(f"### Final Tensile Strength: {final_tensile:.2f}")
    else:
        st.warning("Optimization failed. Adjust constraints for feasibility.")

if __name__ == "__main__":
    main()
