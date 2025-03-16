from src.core.hardware.loihi_driver import LoihiProcessor

# Create and test the processor
processor = LoihiProcessor()
processor.initialize({
    'board_id': 1,
    'chip_id': 0,
    'connection_type': 'local'
})

# Print hardware info
print(processor.get_hardware_info())

# Clean up
processor.shutdown()