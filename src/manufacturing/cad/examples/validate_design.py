from manufacturing.cad.validator import DesignValidator

# Define manufacturing constraints
constraints = {
    "min_wall_thickness": 1.5,  # mm
    "max_dimensions": {
        "x": 800.0,  # mm
        "y": 800.0,  # mm
        "z": 400.0   # mm
    }
}

# Create validator
validator = DesignValidator(constraints)

# Example design data
design_data = {
    "dimensions": {
        "x": 750.0,
        "y": 600.0,
        "z": 300.0
    },
    "min_wall_thickness": 1.2,
    "has_overhangs": True
}

# Validate design
result = validator.validate_design(design_data)

# Check results
if result.is_valid:
    print("Design is valid for manufacturing")
else:
    print("Design validation failed:")
    for issue in result.issues:
        print(f"- {issue['severity'].upper()}: {issue['message']}")