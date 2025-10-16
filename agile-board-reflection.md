📋 **Project Board:** [View Agile Board](https://github.com/users/Amro2023/projects/1)
Agile Project Reflection: Data Analytics for HR Retention at Alpha Manufacturing Solutions

1. Rationale for Decomposition Choices

The project was decomposed into four major epics—Discovery & KPI Design, Data Integration & Build, Testing & Deployment, and Training & Enablement—directly reflecting the RFP deliverables from Alpha Manufacturing Solutions.

Each epic aligns to the consulting proposal’s structure and to the logical data analytics lifecycle:
	•	Phase 1 – Discovery & KPI Design: Focused on defining key HR metrics such as attrition, engagement, and satisfaction. Decomposition into tasks like stakeholder workshops, KPI sign-off, and dashboard wireframing allowed early validation with HR leadership.
	•	Phase 2 – Data Integration & Build: Addressed the RFP’s SAP integration requirement by creating separate tasks for connector configuration, ETL pipeline development, and predictive model creation.
	•	Phase 3 – Testing & Deployment: Mapped to QA, security, and compliance milestones. Tasks ensured performance and data-protection validation before release.
	•	Phase 4 – Training & Strategy Enablement: Translated enablement requirements into deliverables such as documentation, HR workshops, and go-live support.

This structure made dependencies transparent—data integration tasks could proceed in parallel with dashboard design—and aligned sprint cadence (four sprints over sixteen weeks) to tangible outcomes. The approach supports continuous delivery and visibility for both the client and internal stakeholders.

⸻

2. Challenges and Agile Mitigations

Data Security & Access Control

Handling HR data within agile iterations raised privacy and compliance risks. Sprint planning included a dedicated Data Security Definition of Done, ensuring encryption and anonymization were implemented before any data entered testing environments.

SAP System Dependencies

Integration tasks required coordination with the client’s SAP partner. Agile ceremonies (stand-ups, sprint reviews) were used to synchronize schedules, capture blockers early, and adjust sprint scope.

Predictive-Model Validation

Iterative modeling sometimes conflicted with strict delivery timelines. By incorporating model performance reviews into sprint retrospectives, the team could balance innovation with predictability.

HR Stakeholder Engagement

Non-technical users initially struggled with backlog terminology. Conducting joint backlog-refinement sessions translated analytics deliverables into HR-friendly language, improving adoption and feedback cycles.

Overall, agile practices such as incremental delivery, sprint retrospectives, and transparent boards allowed adaptive risk management and constant alignment to business goals.

⸻

3. Visual Evidence of Execution
<img width="1440" height="900" alt="Screenshot 2025-10-16 at 6 29 42 PM" src="https://github.com/user-attachments/assets/da0c4e19-49c9-4dca-86c6-bbb671223ef3" />
<img width="1440" height="900" alt="Screenshot 2025-10-16 at 6 32 19 PM" src="https://github.com/user-attachments/assets/2da9c422-699c-460d-9ca2-aad09ddde7c2" />

<img width="1440" height="900" alt="Screenshot 2025-10-16 at 6 33 44 PM" src="https://github.com/user-attachments/assets/8dd36b97-8cab-49f1-b14b-d811a7961b2e" />


⸻

## 4. Group Contribution p

| **Team Member(s)**         | **Primary Contribution**                                      | **Phase / Epic Focus**                      |
|-----------------------------|---------------------------------------------------------------|---------------------------------------------|
| **Amro Osman & Nick Cantafio** | Project setup, SAP integration architecture, sprint planning  | Phase 2 – Integration & Build               |
| **Ashwabh Dhawan**          | HR metrics design, retention KPI validation                   | Phase 1 – Discovery & KPI Design            |
| **Spencer Chidley & Amro Osman** | Dashboard and predictive-model development                    | Phase 3 – Testing & Deployment              |
| **All Members**             | Documentation, client workshops, go-live training             | Phase 4 – Training & Enablement             |
5. Linked Deliverable
	•	✅ Live HR Attrition Dashboard: https://hr-attrition-dashboard6520.streamlit.app
Demonstrates MVP predictive and visualization capabilities integrated with SAP HR data flows.
