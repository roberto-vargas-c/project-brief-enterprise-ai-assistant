# Retrieval evaluation (offline TF-IDF embeddings + keyword search, k=8, no LLM)

**Recall@k: 100% · Top-1 hit: 93%**

| question | category | recall | top1 |
|---|---|---|---|
| How many days per week are hybrid employees expected to be in the office? | policy | 100% | ✅ |
| What are the core hours when remote employees must be available? | policy | 100% | ✅ |
| How do I request annual leave? | policy | 100% | ✅ |
| Within how many days can a customer return unopened goods? | policy | 100% | ✅ |
| What is the minimum password length? | policy | 100% | ✅ |
| How quickly must a personal data breach be reported to the supervisory authority? | policy | 100% | ✅ |
| What is the deadline for submitting an expense claim? | policy | 100% | ✅ |
| Who needs to approve a contract worth €300,000? | policy | 100% | ✅ |
| What commission accelerator applies above 150% of target? | policy | 100% | ✅ |
| After how many minutes of inactivity must a laptop screen lock? | policy | 100% | ❌ |
| How much does the Nordholm Lysa 3-Seater Sofa cost? | product | 100% | ✅ |
| Is the CoreForge Half Rack 500 in stock, and in which warehouse? | product | 100% | ✅ |
| What did the IronStore Compact Bluetooth Speaker cost in the old 2021 catalog? | trap: outdated asked | 100% | ✅ |
| What do the documents with ID CS-POL-014 cover? | trap: shared ID | 100% | ✅ |
| Which product has the SKU IS-BED-002? | trap: reused SKU | 100% | ✅ |