"""Evaluate configured PPE requirements alongside normal pipeline observations."""

from two_stage_ppe import CompliancePolicy, PPEPipeline


policy = CompliancePolicy.load("examples/compliance_policy.yaml")
pipeline = PPEPipeline("person_model.pt", "ppe_model.pt")
result = pipeline.predict("image.jpg", compliance_policy=policy)

for person in result.persons:
    print(person.id, person.compliance.to_dict() if person.compliance else None)
