"""Run one loaded pipeline across a directory."""

from two_stage_ppe import PPEPipeline

pipeline = PPEPipeline("person.pt", "ppe.pt", person_class="person")
results = pipeline.process("images", "output", save_images=True, save_json=True)
print(f"Processed {len(results)} image(s)")

