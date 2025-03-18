from manufacturing.cad.parametric import UCAVParametricDesign

# Create parametric design instance
design = UCAVParametricDesign()

# Modify parameters
design.set_parameter("wingspan", 14000.0)
design.set_parameter("wing_sweep", 40.0)

# Generate design data
design_data = design.generate_design()

# Access design properties
print(f"Wing Area: {design_data['dimensions']['wing_area']} mm²")
print(f"Effective Wingspan: {design_data['aerodynamics']['effective_wingspan']} mm")