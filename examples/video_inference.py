"""Stream a user-provided video through one loaded pipeline."""

from two_stage_ppe import PPEPipeline

pipeline = PPEPipeline("person.pt", "ppe.pt", ppe_batch_size=8)
summary = pipeline.predict_video(
    "input.mp4",
    "output.mp4",
    save_json=True,
    frame_stride=1,
)
print(summary.to_dict())
