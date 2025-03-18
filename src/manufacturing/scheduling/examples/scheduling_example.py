from manufacturing.scheduling.production_scheduler import (
    ProductionScheduler, ProductionTask, ProductionPriority
)
from manufacturing.scheduling.resource_allocator import (
    ManufacturingResourceAllocator, ResourceRequest, ManufacturingResourceType
)
from manufacturing.scheduling.progress_tracker import ProgressTracker, TaskStatus
from manufacturing.scheduling.quality_control import QualityController, QAStatus
from manufacturing.scheduling.cost_modeling import CostModel
from datetime import datetime, timedelta
import time

# Create scheduler, resource allocator, and progress tracker
scheduler = ProductionScheduler()
resource_allocator = ManufacturingResourceAllocator()
progress_tracker = ProgressTracker()

# Create production tasks
tasks = [
    ProductionTask(
        task_id="wing_assembly",
        description="Assemble main wing components",
        priority=ProductionPriority.HIGH,
        estimated_duration=8.0,
        dependencies=[],
        resources_required={
            "assembly_station": 2,
            "quality_control": 1
        }
    ),
    ProductionTask(
        task_id="fuselage_layup",
        description="Carbon fiber layup for fuselage",
        priority=ProductionPriority.HIGH,
        estimated_duration=12.0,
        dependencies=[],
        resources_required={
            "composite_layup": 2
        }
    ),
    ProductionTask(
        task_id="final_assembly",
        description="Final UCAV assembly",
        priority=ProductionPriority.MEDIUM,
        estimated_duration=16.0,
        dependencies=["wing_assembly", "fuselage_layup"],
        resources_required={
            "assembly_station": 3,
            "quality_control": 2
        }
    )
]

# Add tasks and allocate resources
for task in tasks:
    # Request resources
    for resource_type, quantity in task.resources_required.items():
        request = ResourceRequest(
            resource_type=ManufacturingResourceType(resource_type),
            quantity=quantity,
            duration=timedelta(hours=task.estimated_duration),
            priority=task.priority.value,
            task_id=task.task_id
        )
        if not resource_allocator.request_resources(request):
            print(f"Failed to allocate resources for {task.task_id}")
            break
    else:
        # Add task to scheduler if resource allocation successful
        scheduler.add_task(task)
        # Initialize progress tracking
        progress_tracker.start_task(task.task_id, task.estimated_duration)

# Optimize schedule
scheduler.optimize_schedule()

# Simulate task execution and progress updates
# Add QualityController to existing initializations
quality_controller = QualityController()

# Add import
from manufacturing.scheduling.cost_modeling import CostModel

# Add to initializations
cost_model = CostModel()

# Update simulate_task_execution function
def simulate_task_execution():
    for task in scheduler.get_schedule():
        task_id = task["task_id"]
        print(f"\nExecuting task: {task['description']}")
        
        # Simulate materials used (in real system, this would come from actual usage)
        materials_used = {
            "carbon_fiber": 5.0,    # kg
            "composite_resin": 2.0  # liters
        } if task_id == "fuselage_layup" else {
            "aluminum": 3.0         # kg
        }
        
        # Calculate costs at start
        costs = cost_model.calculate_task_cost(
            task_id=task_id,
            resources_used=task["resources"],
            duration=task["duration"],
            materials_used=materials_used
        )
        
        # Simulate progress updates with QA checks
        for progress in range(0, 101, 20):
            progress_tracker.update_progress(task_id, progress)
            print(f"Progress: {progress}%")
            
            # Perform QA checks at 50% and 100% completion
            if progress in [50, 100] and task_id in quality_controller.checkpoints:
                print(f"\nPerforming QA checks for {task_id}")
                for checkpoint_id, checkpoint in quality_controller.checkpoints[task_id].items():
                    # Simulate measurements (in real system, these would come from sensors)
                    measurements = {
                        measure: 1.0  # Simulated perfect measurements
                        for measure in checkpoint.required_measurements
                    }
                    
                    # Perform QA check
                    result = quality_controller.perform_qa_check(
                        task_id, checkpoint_id, measurements
                    )
                    
                    print(f"QA Check - {checkpoint.name}: {result['status'].value}")
                    if result['status'] == QAStatus.FAILED:
                        print(f"Issues found: {result['issues']}")
                        progress_tracker.mark_failed(task_id, f"Failed QA: {checkpoint.name}")
                        return
            
            time.sleep(1)  # Simulate work being done
            
            # Existing delay simulation
            if progress == 60 and task_id == "fuselage_layup":
                progress_tracker.mark_delayed(task_id)
                print("Task delayed!")
        
        # Complete task and release resources
        progress_tracker.complete_task(task_id)
        resource_allocator.release_resources(task_id)

# Add after existing status displays
print("\nCost Analysis:")
total_costs = cost_model.get_total_production_cost()
print("\nTotal Production Costs:")
for category, cost in total_costs.items():
    print(f"{category.title()}: ${cost:,.2f}")

print("\nPer-Task Cost Breakdown:")
for task_id in progress_tracker.get_all_progress().keys():
    task_costs = cost_model.get_task_cost_summary(task_id)
    if task_costs:
        print(f"\nTask: {task_id}")
        for category, cost in task_costs.items():
            if category != "total":
                print(f"  {category.title()}: ${cost:,.2f}")
        print(f"  Total: ${task_costs['total']:,.2f}")

# Run simulation
simulate_task_execution()

# Display final status
print("\nFinal Production Status:")
for task_id, progress in progress_tracker.get_all_progress().items():
    print(f"\nTask: {task_id}")
    print(f"Status: {progress['status'].value}")
    print(f"Completion: {progress['completion_percentage']}%")
    print(f"Actual Duration: {progress['actual_duration']:.2f} hours")
    if progress['status'] == TaskStatus.DELAYED:
        print("Note: Task was delayed during execution")

# Display resource status
resource_status = resource_allocator.get_resource_status()
print("\nFinal Resource Status:")
for resource_type, status in resource_status.items():
    print(f"\n{resource_type}:")
    print(f"Used: {status['used']}")
    print(f"Available: {status['available']}")
    print(f"Total: {status['total']}")

# Add QA results to final status display
print("\nQuality Assurance Results:")
for task_id in progress_tracker.get_all_progress().keys():
    qa_results = quality_controller.get_qa_results(task_id)
    if qa_results:
        print(f"\nTask: {task_id}")
        for checkpoint_id, result in qa_results.items():
            print(f"Checkpoint: {result['checkpoint']}")
            print(f"Status: {result['status'].value}")
            if result['issues']:
                print(f"Issues: {result['issues']}")