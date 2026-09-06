"""Track workers in a user-provided video without storing frame history."""

from two_stage_ppe import PPEPipeline

pipeline = PPEPipeline("person.pt", "ppe.pt", ppe_batch_size=8)
summary = pipeline.predict_video(
    "input.mp4",
    output_path="output.mp4",
    save_json=True,
    tracking=True,
)
print(summary.to_dict())
