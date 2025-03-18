from manufacturing.training.specialized_pipeline import TrainingDomain
from manufacturing.training.transfer_learning import TransferLearningPipeline, TransferStrategy
import numpy as np

# Create source and target domain data
source_data = {
    "features": np.random.rand(1000, 10),
    "labels": np.random.rand(1000, 1)
}

target_data = {
    "features": np.random.rand(500, 10),
    "labels": np.random.rand(500, 1)
}

# Create transfer learning pipeline
transfer_pipeline = TransferLearningPipeline(
    source_domain=TrainingDomain.AERODYNAMICS,
    target_domain=TrainingDomain.STRUCTURAL,
    strategy=TransferStrategy.ADAPTIVE
)

# Execute transfer learning
results = transfer_pipeline.transfer_and_train(source_data, target_data)

# Check results
if results["status"] == "success":
    print("Transfer learning completed successfully")
    print(f"Features transferred: {results['transfer_stats']['features_transferred']}")
    print(f"Strategy used: {results['transfer_stats']['strategy_used']}")
    print("\nTarget domain results:")
    print(f"Validation metrics: {results['target_results']['validation']}")
else:
    print(f"Transfer learning failed: {results.get('error', 'Unknown error')}")