"""Orchestrate a complete project from a user-provided Pascal VOC dataset."""

from two_stage_ppe import TrainingConfig, train_two_stage

config = TrainingConfig(
    dataset="dataset",
    output="runs/my_project",
    parent_class="person",
    child_classes=["helmet", "gloves", "boots", "vest"],
    person_model="yolo11n.pt",
    ppe_model="yolo11n.pt",
)

result = train_two_stage(config)
print(result.to_dict())

