"""Run the pipeline on one user-provided image."""

from two_stage_ppe import PPEPipeline

pipeline = PPEPipeline("person.pt", "ppe.pt")
result = pipeline.predict("image.jpg")
result.save_image("output/image.jpg")
result.save_json("output/image.json")

